#include "include/server.hpp"
#include "include/handlers.hpp"
#include "include/llama3.hpp"
 
int main() {    
    nn::Parameters parameters; 

    Settings settings {
        .model_dimension=2048,
        .number_of_layers=16,
        .number_of_q_heads=32,
        .number_of_kv_heads=8,
        .vocabulary_size=128256,
        .multiple_of = 256, 
        .norm_epsilon = 1e-5,
        .rope_theta = 500000.0,
        .ffn_dimension_multiplier=1.5,
        .batch_size_limit=8,
        .sequence_length_limit=512
    };
    
    Transformer model(float32, settings); 

    parameters.initialize("../data/llama3-model"); 
    model.initialize(parameters);
     
    Server server(8080);
    while (true) {
        Socket socket = server.accept();   
        try {
            Tensor input = receive(server, socket); 
            int position; server.read(socket, &position, sizeof(int)); 
            std::cout << "Postion: " << position << std::endl;
            Tensor outs = repack(model(input, position)); 
            send(server, socket, outs);
        } 
        
        catch (const std::exception& exception) {
            std::cerr << "Unexpected model error: " << exception.what() << "\n";
            continue;
        } 
    }     
}