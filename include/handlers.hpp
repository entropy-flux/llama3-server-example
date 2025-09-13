#include <tannic.hpp>
#include <tannic/serialization.hpp>
#include "server.hpp"

namespace tannic {
  
void send(Server const& server, Socket& socket, Tensor const& tensor) {
    Header header = headerof(tensor);
    Metadata<Tensor> metadata = metadataof(tensor);
    server.write(socket, &header, sizeof(Header));
    server.write(socket, &metadata, sizeof(Metadata<Tensor>)); 
    server.write(socket, tensor.shape().address(), tensor.shape().rank() * sizeof(size_t));
    server.write(socket, tensor.bytes(), tensor.nbytes());
}

Tensor receive(Server const& server, Socket& socket) {
    Header header{};
    Metadata<Tensor> metadata{};  
    bool success;
    
    success = server.read(socket, &header, sizeof(Header));
    if (!success) {
        throw Exception("Issues reading header from socket");
    } 

    else if(header.magic != MAGIC) {
        throw Exception("Invalid magic! Closing connection.\n");
    }

    success = server.read(socket, &metadata, sizeof(Metadata<Tensor>));
    if (!success) {
        throw Exception("Issues reading metadata from socket");
    } 
 
    Shape shape;  
    for (uint8_t dimension = 0; dimension < metadata.rank; dimension++) {
        size_t size; 
        success = server.read(socket, &size, sizeof(size_t));
        if (!success) {
            throw Exception("Client disconnected.");
        }
        shape.expand(size);
    }
 
    std::shared_ptr<Buffer> buffer = std::make_shared<Buffer>(metadata.nbytes);
    success = server.read(socket, buffer->address(), metadata.nbytes);
    if (!success) {
        throw Exception("Client disconnected.");
    } 
    return Tensor(dtypeof(metadata.dcode), shape, 0, buffer);  
}

}