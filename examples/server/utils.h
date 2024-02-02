#include <iostream>
#include <deque>
#include <thread>

#include "json.hpp"

using namespace std;
using json = nlohmann::json;

struct ServerTask {
    enum TaskType {
        TASK_COMPLETION = 0,
        TASK_CHAT_COMPLETION
    };

    int _id;
    json _data;
    TaskType _type;

    ServerTask(int id, json data, TaskType taskType) {
        _id = id;
        _data = data;
        _type = taskType;
    }

    void dump() {
        cout << "request task id:" << _id << endl;
    }
};

class ServerRequestTaskQueue {
    mutex _m;
    condition_variable _cv;
    deque<ServerTask> _tasks;

    unsigned int _id = 0;

    public:
        int push(json &data, ServerTask::TaskType taskType) {
            lock_guard lk(_m);
            int taskId = _id++;
            _tasks.push_back(ServerTask(taskId, data, taskType));

            _cv.notify_one();

            return taskId;
        }

        ServerTask pop() {
            unique_lock lk(_m);
            _cv.wait(lk);
            ServerTask task = _tasks.front();
            _tasks.pop_front();

            return task;
        }
};

class ServerResponseTaskQueue {
    mutex _m;
    condition_variable _cv;
    deque<ServerTask> _tasks;

    public:
        bool push(ServerTask task) {
            {
                lock_guard lk(_m);
                _tasks.push_back(task);
            }
            _cv.notify_all();

            return true;
        }

        json result(int taskId) {
            while (true) {
                unique_lock lk(_m);
                _cv.wait(lk);

                for (deque<ServerTask>::iterator it = _tasks.begin();
                     it != _tasks.end(); it++) {
                    if ((*it)._id == taskId) {
                        ServerTask task = *it;
                        _tasks.erase(it);
                        return task._data;
                    }
                }
            }
        }
};
