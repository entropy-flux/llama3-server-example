#include <cmath>
#include <tannic.hpp>
#include <tannic/filter.hpp>
#include <tannic-nn.hpp>
#include <tannic-nn/functional.hpp>  

using namespace tannic;

struct Settings {
    size_t model_dimension;
    size_t number_of_layers;
    size_t number_of_q_heads;
    size_t number_of_kv_heads;
    int vocabulary_size;
    int multiple_of; 
    float norm_epsilon; 
    float rope_theta;
    float ffn_dimension_multiplier;
    bool use_scaled_rope;
    size_t batch_size_limit;
    size_t sequence_length_limit;
};

struct RMS : public nn::Module { 
    nn::Parameter weight;
    float epsilon;

    constexpr RMS(type dtype, size_t dimension, float epsilon) 
    :   weight(dtype, {dimension}) 
    ,   epsilon(epsilon){}  

    Tensor forward(Tensor const& input) const { 
        auto norm = input * rsqrt(mean(input*input, -1, true), epsilon);
        return weight * norm;
    }

    void initialize(std::string const& name, nn::Parameters& parameters) const {
        weight.initialize(name + ".weight", parameters);
    }
}; 

Tensor split(Tensor sequence, int number_of_heads) {    
    int batch_size = sequence.size(0);
    int sequence_length = sequence.size(1);
    int model_dimension = sequence.size(2);  
    sequence = sequence.view(batch_size, sequence_length, number_of_heads, model_dimension / number_of_heads);
    return repack(sequence.transpose(1, 2));
}

Tensor merge(Tensor sequence) {  
    int batch_size = sequence.size(0);
    int number_of_heads = sequence.size(1);
    int sequence_lenght = sequence.size(2);
    int heads_dimension = sequence.size(3);
    sequence = sequence.transpose(1, 2);
    return reshape(sequence, batch_size, sequence_lenght, heads_dimension* number_of_heads);
}

Tensor compute_frequencies(type dtype, size_t model_dimension, size_t sequence_length_limit, double theta = 10000.0) {
    auto scale = std::log(theta) / model_dimension;
    Tensor rho = ones(dtype, {sequence_length_limit, model_dimension / 2});
    Tensor phi(dtype, {sequence_length_limit, model_dimension / 2}); 
    for(auto position = 0; position < sequence_length_limit; position++) {
        for(auto dimension = 0; dimension < model_dimension / 2; dimension++) { 
            phi[position, dimension] = position * std::exp(-2 * dimension * scale); 
        }
    } 
    return polar(rho, phi);
}  
 
Tensor embed_frequencies(Tensor sequence, Tensor frequencies) {   
    int batch_size = sequence.size(0);
    int number_of_heads = sequence.size(1);
    int sequence_length = sequence.size(2);
    int heads_dimension = sequence.size(3);      
    sequence = repack(sequence.view(batch_size, number_of_heads, sequence_length, heads_dimension / 2, 2));   
    sequence = complexify(sequence);  
    sequence = sequence * frequencies; 
    sequence = realify(sequence); 
    return sequence.view(batch_size, number_of_heads, sequence_length, heads_dimension);
}

Tensor scaled_dot_attention(Tensor query, Tensor key, Tensor value) {  
    float scale = 1 / std::sqrt(value.size(-1));
    auto score = matmul(query, key.transpose(2, 3)) * scale;
    Tensor output = nn::softmax(score, -1);
    return matmul(output, value);
} 

Tensor scaled_dot_attention(Tensor query, Tensor key, Tensor value, Tensor mask) {  
    float scale = 1 / std::sqrt(value.size(-1));
    auto score = matmul(query, key.transpose(2, 3)) * scale + mask;
    Tensor output = nn::softmax(score, -1); 
    return matmul(output, value);
} 

struct Cache : nn::Module {
    Tensor buffer;

    Cache(
        type dtype,
        size_t batch_size_limit,
        size_t sequence_length_limit,
        size_t number_of_heads,
        size_t heads_dimension
    ) {
        buffer = zeros(dtype, {
            batch_size_limit, 
            sequence_length_limit, 
            number_of_heads, 
            heads_dimension
        });
    }

    Tensor forward(Tensor sequence, int position) {
        int batch_size = sequence.size(0);
        int sequence_length = sequence.size(2); 
        buffer[{0, batch_size}][{position, position + sequence_length}] = sequence.transpose(1, 2);
        sequence = transpose(buffer[{0, batch_size}][{0, position + sequence_length}], 1, 2); 
        return repack(sequence);
    }
};


struct Attention : nn::Module {
    size_t number_of_q_heads;
    size_t number_of_kv_heads;
    size_t number_of_repeats;
    size_t heads_dimension;

    nn::Linear q_projector;
    nn::Linear k_projector;
    nn::Linear v_projector;
    nn::Linear o_projector;

    Cache k_cache;
    Cache v_cache;
 
    Attention(type dtype, Settings settings) 
    :   number_of_q_heads(settings.number_of_q_heads)
    ,   number_of_kv_heads(settings.number_of_kv_heads)
    ,   heads_dimension(settings.model_dimension / number_of_q_heads)
    ,   q_projector(dtype, settings.model_dimension, heads_dimension * number_of_q_heads , false)
    ,   k_projector(dtype, settings.model_dimension, heads_dimension * number_of_kv_heads, false)
    ,   v_projector(dtype, settings.model_dimension, heads_dimension * number_of_kv_heads, false)
    ,   number_of_repeats(number_of_q_heads / number_of_kv_heads)
    ,   o_projector(dtype, number_of_q_heads * heads_dimension, settings.model_dimension, false)
    ,   k_cache(dtype, settings.batch_size_limit, settings.sequence_length_limit, number_of_kv_heads, heads_dimension)
    ,   v_cache(dtype, settings.batch_size_limit, settings.sequence_length_limit, number_of_kv_heads, heads_dimension)
    {}
 
    void initialize(nn::Parameters& parameters) const {
        q_projector.initialize("q_projector", parameters);
        k_projector.initialize("k_projector", parameters);
        v_projector.initialize("v_projector", parameters);
        o_projector.initialize("o_projector", parameters);
    }

    void initialize(std::string const& name, nn::Parameters& parameters) const {
        q_projector.initialize(name + ".q_projector", parameters);
        k_projector.initialize(name + ".k_projector", parameters);
        v_projector.initialize(name + ".v_projector", parameters);
        o_projector.initialize(name + ".o_projector", parameters);
    } 

    Tensor forward(Tensor sequence, int position, Tensor frequencies) {  
        Tensor query = split(q_projector(sequence), number_of_q_heads); 
        Tensor key   = split(k_projector(sequence), number_of_kv_heads);  
        Tensor value = split(v_projector(sequence), number_of_kv_heads);    
 
        query = embed_frequencies(query, frequencies); 
        key   = embed_frequencies(key, frequencies);    
   
        key   = repeat(k_cache(key,   position), number_of_repeats, 1);    
        value = repeat(v_cache(value, position), number_of_repeats, 1);
        auto score = scaled_dot_attention(query, key, value);
        return o_projector(merge(score));
    }

    Tensor forward(Tensor sequence, int position, Tensor frequencies, Tensor mask) {  
        Tensor query = split(q_projector(sequence), number_of_q_heads); 
        Tensor key   = split(k_projector(sequence), number_of_kv_heads);  
        Tensor value = split(v_projector(sequence), number_of_kv_heads);    
 
        query = embed_frequencies(query, frequencies); 
        key   = embed_frequencies(key, frequencies);    
   
        key   = repeat(k_cache(key,   position), number_of_repeats, 1);    
        value = repeat(v_cache(value, position), number_of_repeats, 1);
        auto score = scaled_dot_attention(query, key, value, mask);
        return o_projector(merge(score));
    }
};  
 
struct FFN : nn::Module {
    nn::Linear input_layer;
    nn::Linear output_layer;
    nn::Linear gate_layer;

    FFN(type dtype, size_t model_dimension, size_t hidden_dimension) 
    :   input_layer(dtype, model_dimension, hidden_dimension, false) 
    ,   output_layer(dtype, hidden_dimension, model_dimension, false)
    ,   gate_layer(dtype, model_dimension, hidden_dimension, false)
    {}

    Tensor forward(Tensor features) {
        features = nn::silu(input_layer(features)) * gate_layer(features);
        return output_layer(features);
    }

    void initialize(std::string const& name, nn::Parameters& parameters) const {
        input_layer.initialize(name + ".input_layer", parameters);
        output_layer.initialize(name + ".output_layer", parameters);
        gate_layer.initialize(name + ".gate_layer", parameters); 
    }
};

constexpr auto scale_ffn(size_t dimension, float dimension_multiplier, int multiple_of) {
    dimension = size_t(2 * dimension / 3);
    dimension = size_t(dimension * dimension_multiplier);
    return multiple_of * ((dimension + multiple_of - 1) / multiple_of); 
}

struct Decoder : nn::Module {
    Attention attention; 
    RMS attention_norm;
    FFN ffn;
    RMS ffn_norm;

    Decoder(type dtype, Settings settings) 
    :   attention(dtype, settings)
    ,   attention_norm(dtype, settings.model_dimension, settings.norm_epsilon)
    ,   ffn(dtype, settings.model_dimension, scale_ffn(4 * settings.model_dimension, settings.ffn_dimension_multiplier, settings.multiple_of))
    ,   ffn_norm(dtype, settings.model_dimension, settings.norm_epsilon) {}

    void initialize(nn::Parameters& parameters) const {
        attention.initialize("attention", parameters);
        attention_norm.initialize("attention_norm", parameters);
        ffn.initialize("ffn", parameters);
        ffn_norm.initialize("ffn_norm", parameters);
    } 

    void initialize(std::string const& name, nn::Parameters& parameters) const {
        attention.initialize(name + ".attention", parameters);
        attention_norm.initialize(name + ".attention_norm", parameters);
        ffn.initialize(name + ".ffn", parameters);
        ffn_norm.initialize(name + ".ffn_norm", parameters);
    } 

    Tensor forward(Tensor sequence, int position, Tensor frequencies) { 
        sequence = attention(attention_norm(sequence), position, frequencies) + sequence;
        return ffn(ffn_norm(sequence)) + sequence;
    }

    Tensor forward(Tensor sequence, int position, Tensor frequencies, Tensor mask) { 
        sequence = attention(attention_norm(sequence), position, frequencies, mask) + sequence;
        return ffn(ffn_norm(sequence)) + sequence;
    }
}; 

struct Transformer : nn::Module {
    type dtype;
    Tensor frequencies; 
    nn::Embedding embeddings;
    nn::List<Decoder> decoders; 
    nn::Linear head;
    RMS norm;

    Transformer(type dtype, Settings settings) 
    :   dtype(dtype) 
    ,   embeddings(dtype, settings.vocabulary_size, settings.model_dimension)
    ,   norm(dtype, settings.model_dimension, settings.norm_epsilon)
    ,   head(dtype, settings.model_dimension, settings.vocabulary_size, false)
    {
        frequencies = compute_frequencies(
            float32, 
            settings.model_dimension / settings.number_of_q_heads, 
            settings.sequence_length_limit * 2,
            settings.rope_theta
        );

        for (auto index = 0; index < settings.number_of_layers; index++) {
            decoders.add(Decoder(dtype, settings));
        }
    } 

    void initialize(nn::Parameters& parameters) const {
        size_t index = 0;
        embeddings.initialize("embeddings", parameters);
        for(auto& decoder: decoders) {
            decoder.initialize("layers." + std::to_string(index), parameters);
            index++;
        } 
        norm.initialize("norm", parameters);
        head.initialize("head", parameters);
    } 

    Tensor forward(Tensor tokens, int position) { 
        size_t sequence_length = tokens.size(1); 
        Tensor sequence = embeddings(tokens);

        if (sequence_length > 1) {
            Tensor mask;
            mask = -infinity(dtype, {size_t{sequence_length}, size_t{sequence_length}});
            mask = triangular(mask, Position::Upper, 1);
            mask = concatenate(zeros(dtype, {size_t{sequence_length}, size_t(position)}), mask, 1);  
            
            for (auto& decoder: decoders) { 
                sequence = decoder(sequence, position, frequencies[{position, position + int(sequence_length)}], mask);
            }                
        } 
        
        else {
            for (auto& decoder: decoders) { 
                sequence = decoder(sequence, position, frequencies[{position, position + int(sequence_length)}]);
            }                
        }
        sequence = norm(sequence); 
        return head(sequence);
    }
};    