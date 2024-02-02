#include <iostream>
#include <thread>

#include <absl/flags/flag.h>
#include <absl/flags/parse.h>
#include <absl/strings/str_split.h>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>

#include "httplib.h"
#include "json.hpp"
#include "utils.h"
#include "../../chatglm.h"

using namespace std;
using namespace httplib;

using json = nlohmann::json;

struct ServerConfig {
    string _host;
    int _port;
    string _model_file;
    int _max_length;
    int _max_context_length;
    int _top_k;
    float _top_p;
    float _temp;
    float _repeat_penalty;
    int _threads;

    ServerConfig(string host, string model, 
                 int max_length, int max_context_length, 
                 int top_k, float top_p, float temp, float repeat_penalty, int threads) {
        pair<string, string> h = absl::StrSplit(host, ':');
        _host = h.first;
        _port = h.second == "" ? 8080 : atoi(h.second.c_str());
        _model_file = model;
        _max_length = max_length;
        _max_context_length = max_context_length;
        _top_k = top_k;
        _top_p = top_p;
        _temp = temp;
        _repeat_penalty = repeat_penalty;
        _threads = threads;
    }

    void dump() {
        cout << "config host: " << _host << endl;
        cout << "config port: " << _port << endl;
        cout << "config model: " << _model_file << endl;
        cout << "config max_length: " << _max_length << endl;
        cout << "config max_context_length: " << _max_context_length << endl;
        cout << "config top_k: " << _top_k << endl;
        cout << "config top_p: " << _top_p << endl;
        cout << "config temp: " << _temp << endl;
        cout << "config repeat_penalty: " << _repeat_penalty << endl;
        cout << "config threads: " << _threads << endl;
    }
};

ABSL_FLAG(string, host, "127.0.0.1:8080", "ip:port");
ABSL_FLAG(string, model, "models/chatglm3-6b-q4_0.bin", "model file");
ABSL_FLAG(int16_t, max_length, 2048, "max total length including prompt and output");
ABSL_FLAG(int16_t, max_context_length, 512, "max context length");
ABSL_FLAG(int16_t, top_k, 0, "top-k sampling");
ABSL_FLAG(float, top_p, 0.7, "top-p sampling");
ABSL_FLAG(float, temp, 0.95, "temperature");
ABSL_FLAG(float, repeat_penalty, 1.0, "penalize repeat sequence of tokens");
ABSL_FLAG(int16_t, threads, 0, "number of threads for inference");

std::string dump_headers(const Headers &headers) {
    std::string s;
    char buf[BUFSIZ];

    for (auto it = headers.begin(); it != headers.end(); ++it) {
        const auto &x = *it;
        snprintf(buf, sizeof(buf), "%s: %s\n", x.first.c_str(), x.second.c_str());
        s += buf;
    }

    return s;
}

std::string log(const Request &req, const Response &res) {
    std::string s;
    char buf[BUFSIZ];
    
    s += "================================\n";  
    snprintf(buf, sizeof(buf), "%s %s %s", req.method.c_str(),
             req.version.c_str(), req.path.c_str());
    s += buf;
    
    std::string query;
    for (auto it = req.params.begin(); it != req.params.end(); ++it) {
      const auto &x = *it;
      snprintf(buf, sizeof(buf), "%c%s=%s",
               (it == req.params.begin()) ? '?' : '&', x.first.c_str(),
               x.second.c_str());
      query += buf;
    }
    snprintf(buf, sizeof(buf), "%s\n", query.c_str());
    s += buf;   
    s += dump_headers(req.headers);
    
    s += "--------------------------------\n";
    snprintf(buf, sizeof(buf), "%d %s\n", res.status, res.version.c_str());
    s += buf;
    s += dump_headers(res.headers);
    s += "\n";  
    if (!res.body.empty()) { s += res.body; }   
    s += "\n";
    
    return s;
}

int start_loop(ServerConfig &conf, chatglm::Pipeline &pl, 
               ServerRequestTaskQueue &request_task_queue, 
               ServerResponseTaskQueue &response_task_queue) {    
    auto perf_streamer = make_shared<chatglm::PerfStreamer>();

    for (;;) {
        ServerTask task = request_task_queue.pop();
        task.dump();

        if (task._type == ServerTask::TASK_COMPLETION) {
            int max_tokens = conf._max_length;
            if (!task._data["max_tokens"].is_null()) max_tokens = task._data["max_tokens"];
            int top_k = conf._top_k;
            if (!task._data["n"].is_null()) top_k = task._data["n"];
            float temperature = conf._temp;
            if (!task._data["temperature"].is_null()) temperature = task._data["temperature"];
            float top_p = conf._top_p;
            if (!task._data["top_p"].is_null()) top_p = task._data["top_p"];

            chatglm::GenerationConfig gen_config(max_tokens, -1, conf._max_context_length,
                                                 temperature > 0, top_k, top_p, temperature, 
                                                 conf._repeat_penalty, conf._threads);

            string prompt = task._data["prompt"];
            string content = pl.generate(prompt, gen_config, perf_streamer.get());

            json response_body;
            response_body["text"] = content;
            response_task_queue.push(ServerTask(task._id, response_body, task._type));

            perf_streamer->reset();
        } else if (task._type == ServerTask::TASK_CHAT_COMPLETION) {
            int max_tokens = conf._max_length;
            if (!task._data["max_tokens"].is_null()) max_tokens = task._data["max_tokens"];
            int top_k = conf._top_k;
            if (!task._data["n"].is_null()) top_k = task._data["n"];
            float temperature = conf._temp;
            if (!task._data["temperature"].is_null()) temperature = task._data["temperature"];
            float top_p = conf._top_p;
            if (!task._data["top_p"].is_null()) top_p = task._data["top_p"];

            chatglm::GenerationConfig gen_config(max_tokens, -1, conf._max_context_length,
                                                 temperature > 0, top_k, top_p, temperature, 
                                                 conf._repeat_penalty, conf._threads);

            vector<chatglm::ChatMessage> messages;

            for (int i=0; i<task._data["messages"].size(); i++) {
                string role = task._data["messages"][i]["role"];
                string prompt = task._data["messages"][i]["content"];
                messages.push_back(chatglm::ChatMessage(role, prompt));
            }

/*
            std::cout << "max_length: " << gen_config.max_length << std::endl;
            std::cout << "max_new_tokens: " << gen_config.max_new_tokens << std::endl;
            std::cout << "max_context_length: " << gen_config.max_context_length << std::endl;
            std::cout << "do_sample: " << gen_config.do_sample << std::endl;
            std::cout << "top_k: " << gen_config.top_k << std::endl;
            std::cout << "top_p: " << gen_config.top_p << std::endl;
            std::cout << "temperature: " << gen_config.temperature << std::endl;
            std::cout << "repetition_penalty: " << gen_config.repetition_penalty << std::endl;
            std::cout << "num_threads: " << gen_config.num_threads << std::endl;

            for (int i=0; i<messages.size(); i++) {
                std::cout << "role: " << messages[i].role << std::endl;
                std::cout << "content: " << messages[i].content << std::endl;
            }
*/
            chatglm::ChatMessage output = pl.chat(messages, gen_config, perf_streamer.get());

            json response_body;
            response_body["role"] = output.role;
            response_body["content"] = output.content;
            response_task_queue.push(ServerTask(task._id, response_body, task._type));

            perf_streamer->reset();
        }
    }
    
    return 0;
}

int main(int argc, char * argv[]) {
    absl::ParseCommandLine(argc, argv);

    ServerConfig conf(absl::GetFlag(FLAGS_host), absl::GetFlag(FLAGS_model),
                      absl::GetFlag(FLAGS_max_length), absl::GetFlag(FLAGS_max_context_length),
                      absl::GetFlag(FLAGS_top_k), absl::GetFlag(FLAGS_top_p),
                      absl::GetFlag(FLAGS_temp), absl::GetFlag(FLAGS_repeat_penalty),
                      absl::GetFlag(FLAGS_threads));
    conf.dump();

    chatglm::Pipeline pl(conf._model_file);
    cout << "load model ok." << endl;

    Server svr;
    ServerRequestTaskQueue request_task_queue;
    ServerResponseTaskQueue response_task_queue;

    if (!svr.is_valid()) {
        cout << "server has an error..." << endl;
        return -1;
    }

    svr.Get("/", [](const Request & /* req */, Response &res) {
        res.set_content("hello world!", "text/plain");
    });

    svr.Post("/v1/completions", [&](const Request &req, Response &res){
        cout << req.body << endl;
        json data = json::parse(req.body);

        int taskId = request_task_queue.push(data, ServerTask::TASK_COMPLETION);
        json result = response_task_queue.result(taskId);

        json response_body;
        boost::uuids::random_generator gen;
        response_body["id"] = boost::uuids::to_string(gen());
        response_body["choices"][0]["text"] = result["text"];
        res.set_content(response_body.dump(), "application/json");
    });

    svr.Post("/v1/chat/completions", [&](const Request &req, Response &res) {
        cout << req.body << endl;
        json data = json::parse(req.body);

        int taskId = request_task_queue.push(data, ServerTask::TASK_CHAT_COMPLETION);
        json result = response_task_queue.result(taskId);

        json response_body;
        boost::uuids::random_generator gen;
        response_body["id"] = boost::uuids::to_string(gen());
        response_body["choices"].push_back({"message", result});
        res.set_content(response_body.dump(), "application/json");
    });

    svr.set_error_handler([](const Request & /* req */, Response &res) {
        const char * fmt = "<p>Error Status: <span style='color:red;'>%d</span></p>";
        char buf[BUFSIZ];
        snprintf(buf, sizeof(buf), fmt, res.status);
        res.set_content(buf, "text/html");
    });

    svr.set_logger([](const Request &req, const Response &res) {
        cout << log(req, res) << endl;
    });

    if (!svr.bind_to_port(conf._host, conf._port)) {
        cout << "fail to bind." << endl;
        return -1;
    }

    cout << "server listen on " << conf._host << ":" << conf._port << endl;

    thread t([&] {
        svr.listen_after_bind();
        return 0;
    });

    start_loop(conf, pl, request_task_queue, response_task_queue);

    t.join();

    return 0;
}