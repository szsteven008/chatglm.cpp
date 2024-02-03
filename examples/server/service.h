#ifndef _SERVICE_H
#define _SERVICE_H

#include "backend.pb.h"
#include "backend.grpc.pb.h"
#include "utils.h"

using namespace backend;
using namespace grpc;

class BackendServiceImpl: public Backend::Service {
    ServerRequestTaskQueue * _request_task_queue;
    ServerResponseTaskQueue * _response_task_queue;

public:
    BackendServiceImpl(ServerRequestTaskQueue * request_task_queue, 
                       ServerResponseTaskQueue * response_task_queue);
    ~BackendServiceImpl();

public:
    grpc::Status Health(ServerContext* context, 
                        const HealthMessage* request, 
                        Reply* reply);

    grpc::Status Predict(ServerContext* context, 
                         const PredictOptions* request, 
                         Reply* response);
};

#endif
