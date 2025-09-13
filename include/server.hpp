#pragma once
#include <unistd.h>
#include <netinet/in.h>
#include <stdexcept>
#include <iostream>
#include <sstream>
#include <string>
#include <cstring>
#include <vector>
#include <cstdint>
#include <memory>
#include <vector>
#include <unordered_map>
#include <arpa/inet.h>  

namespace tannic {

class Endpoint {
public:
    Endpoint() {
        std::memset(&storage_, 0, sizeof(storage_));
        addr4_.sin_family = AF_INET;
        addr4_.sin_addr.s_addr = INADDR_ANY;
        addr4_.sin_port = htons(0);
        length_ = sizeof(sockaddr_in);
    }

    Endpoint(uint16_t port, in_addr_t address = INADDR_ANY) {
        std::memset(&storage_, 0, sizeof(storage_));
        addr4_.sin_family = AF_INET;
        addr4_.sin_addr.s_addr = address;
        addr4_.sin_port = htons(port);
        length_ = sizeof(sockaddr_in);
    }

    sockaddr* address() {
        return reinterpret_cast<sockaddr*>(&storage_);
    }

    socklen_t* length() {
        return &length_;
    }

private:
    union {
        sockaddr_storage storage_;
        sockaddr_in addr4_;
        sockaddr_in6 addr6_;
    };
    socklen_t length_{};
}; 

class Socket {
public:
    Socket() {
        descriptor_ = ::socket(AF_INET, SOCK_STREAM, 0);
        if (descriptor_ < 0)
            throw std::runtime_error("socket() failed");
    }

    Socket(Socket&& other) noexcept : descriptor_(other.descriptor_) {
        other.descriptor_ = -1;
    }

    Socket& operator=(Socket&& other) noexcept {
        if (this != &other) {
            close();
            descriptor_ = other.descriptor_;
            other.descriptor_ = -1;
        }
        return *this;
    }

    ~Socket() { close(); }

    void reuse() {
        int option = 1;
        auto status = setsockopt(descriptor_, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &option, sizeof(option));
        if (status < 0)
            throw std::runtime_error("setsockopt() failed");
    }

    void bind(uint16_t port, in_addr_t address = INADDR_ANY) {
        Endpoint endpoint(port, address);
        auto status = ::bind(descriptor_, endpoint.address(), *endpoint.length());
        if (status < 0)
            throw std::runtime_error("bind() failed");
    }

    void listen(int backlog = 3) {
        if (::listen(descriptor_, backlog) < 0)
            throw std::runtime_error("listen() failed");
    }

    Socket accept() {
        Endpoint client;
        int descriptor = ::accept(descriptor_, client.address(), client.length());
        if (descriptor < 0)
            throw std::runtime_error("accept() failed");
        return Socket(descriptor);
    }

    ssize_t send(const void* buffer, size_t length, int flags = 0) {
        return ::send(descriptor_, buffer, length, flags);
    }

    ssize_t receive(void* buffer, size_t length, int flags = 0) {
        return ::recv(descriptor_, buffer, length, flags);
    }

private:
    explicit Socket(int descriptor) : descriptor_(descriptor) {}

    void close() {
        if (descriptor_ >= 0) {
            ::close(descriptor_);
            descriptor_ = -1;
        }
    }

    int descriptor_{-1};
}; 
 
class Server {   
public: 
    Server(int port) : listener_(), port_(port) {
        listener_.reuse();
        listener_.bind(port_);
        listener_.listen();
        std::cout << "Listening on port " << port_ << "...\n";
    }

    Socket accept() {
        return listener_.accept();    
    }

    bool read(Socket& socket, void* buffer, size_t nbytes) const {
        size_t total = 0;
        while (total < nbytes) {
            ssize_t n = socket.receive(static_cast<char*>(buffer) + total, nbytes - total);
            if (n == 0) {
                return false;
            }
            if (n < 0) {
                // real error
                throw std::runtime_error("Error while reading from socket");
            }
            total += n;
        }
        return true;
    }

    bool write(Socket& socket, const void* buffer, size_t nbytes) const {  
        size_t total = 0;
        while (total < nbytes) {
            ssize_t n = socket.send(static_cast<const char*>(buffer) + total, nbytes - total);
            if (n == 0) {
                return false;
            }
            if (n < 0) {
                throw std::runtime_error("Error while writing to socket");
            }
            total += n;
        }
        return true;
    }


private:
    int port_;
    Socket listener_;  
};

} // namespace tannic