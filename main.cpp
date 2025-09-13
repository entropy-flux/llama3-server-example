#include "include/server.hpp"
#include "include/handlers.hpp"
#include "include/llama3.hpp"
 
int main() {    
    nn::Parameters parameters; 

    Settings settings{
        .model_dimension=16,
        .number_of_layers=3,
        .number_of_q_heads=4,
        .number_of_kv_heads=2,
        .vocabulary_size=128,
        .multiple_of = 256, 
        .norm_epsilon = 1e-5,
        .rope_theta = 500000,
        .ffn_dimension_multiplier=1.5,
        .batch_size_limit=4,
        .sequence_length_limit=16
    };
    
    Transformer model(float32, settings); 

    parameters.initialize("../dev-transformer"); 
    model.initialize(parameters);
     
    Server server(8080);
    while (true) {
        Socket socket = server.accept();   
        try {
            Tensor input = receive(server, socket); 
            int position; server.read(socket, &position, sizeof(int)); 
            Tensor outs = repack(model(input, position));
            send(server, socket, outs);
        } 
        
        catch (const Exception& exception) {
            std::cerr << "Unexpected model error: " << exception.what() << "\n";
            break;
        } 
    }     
}