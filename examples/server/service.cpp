#include "service.h"

using namespace std;

BackendServiceImpl::BackendServiceImpl(ServerRequestTaskQueue * request_task_queue, 
                                       ServerResponseTaskQueue * response_task_queue) : 
                                       _request_task_queue(request_task_queue), 
                                       _response_task_queue(response_task_queue) {
}

BackendServiceImpl::~BackendServiceImpl() {
}

grpc::Status BackendServiceImpl::Health(ServerContext* context, 
                    const HealthMessage* request, 
                    Reply* response) {
    // Implement Health RPC
    response->set_message("OK");
    return Status::OK;
}

grpc::Status BackendServiceImpl::Predict(ServerContext* context, 
                                         const PredictOptions* request, 
                                         Reply* response) {
    cout << "predict prompt: " << request->prompt() << endl;

    json message = { {"role", "user"}, {"content", request->prompt()} };
    json data;
    data["messages"].push_back(message);
    if (request->tokens() > 0) data["max_tokens"] = request->tokens();
    if (request->topk() > 0) data["n"] = request->topk();
    if (request->temperature() > 0) data["temperature"] = request->temperature();
    if (request->topp() > 0) data["top_p"] = request->topp();

    int taskId = _request_task_queue->push(data, ServerTask::TASK_CHAT_COMPLETION);
    json result = _response_task_queue->result(taskId);
    json response_body;

    cout << "result role:" << result["role"] << " result content: " << result["content"] << endl;

    response->set_message(result["content"]);

    return grpc::Status::OK;
}
