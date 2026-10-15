#pragma once

#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>



















class KittensClub {
public:
    __host__ inline KittensClub(const int *device_ids, const int num_devices);
    __host__ inline KittensClub(const int *device_ids, const cudaStream_t *streams, const int num_devices);
    __host__ inline ~KittensClub();


    __host__ inline void execute(std::function<void(int, cudaStream_t)> task);

private:

    bool stop;
    std::vector<bool> task_available;
    int n_task_done;


    std::vector<std::thread> workers;


    std::vector<cudaStream_t> streams;


    __host__ inline void worker(int worker_id, int device_id);


    std::function<void(int, cudaStream_t)> current_task;


    std::mutex mutex;
    std::condition_variable cond_task_available;
    std::condition_variable cond_task_done;
};

__host__ inline KittensClub::KittensClub(const int *device_ids, const int num_devices) : stop(false), n_task_done(0) {
    for (size_t dev_idx = 0; dev_idx < num_devices; ++dev_idx) {
        task_available.push_back(false);
        streams.push_back(0);
        workers.emplace_back([this, dev_idx, device_ids] { worker(dev_idx, device_ids[dev_idx]); });
    }
}

__host__ inline KittensClub::KittensClub(const int *device_ids, const cudaStream_t *streams_in, const int num_devices) : stop(false), n_task_done(0) {
    for (size_t dev_idx = 0; dev_idx < num_devices; ++dev_idx) {
        task_available.push_back(false);
        streams.push_back(streams_in[dev_idx]);
        workers.emplace_back([this, dev_idx, device_ids] { worker(dev_idx, device_ids[dev_idx]); });
    }
}

__host__ inline KittensClub::~KittensClub() {
    {
        std::lock_guard<std::mutex> lock(mutex);
        stop = true;
    }
    cond_task_available.notify_all();
    for (std::thread &worker : workers) {
        worker.join();
    }
}

__host__ inline void KittensClub::execute(std::function<void(int, cudaStream_t)> task) {
    {
        std::lock_guard<std::mutex> lock(mutex);
        current_task = task;
        for (size_t i = 0; i < task_available.size(); ++i)
            task_available[i] = true;
    }
    cond_task_available.notify_all();
    {
        std::unique_lock<std::mutex> lock(mutex);
        cond_task_done.wait(lock, [this] { return n_task_done == workers.size(); });
        n_task_done = 0;
    }
}

__host__ inline void KittensClub::worker(int worker_id, int device_id) {
    cudaSetDevice(device_id);
    while (true) {
        std::function<void(int, cudaStream_t)> task;
        {
            std::unique_lock<std::mutex> lock(mutex);
            cond_task_available.wait(lock, [this, worker_id] { return stop || task_available[worker_id]; });

            if (stop)
                return;

            task = current_task;
            task_available[worker_id] = false;
        }
        task(worker_id, streams[worker_id]);
        {
            std::lock_guard<std::mutex> lock(mutex);
            ++n_task_done;
            if (n_task_done == workers.size())
                cond_task_done.notify_one();
        }
    }
}
