#pragma once

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <exception>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace cpso {

// Small persistent CPU thread pool. parallel_for schedules a bounded number
// of worker tasks; each task claims indices dynamically from an atomic counter.
// The supplied callable must be safe for concurrent invocation.
class ThreadPool {
public:
    explicit ThreadPool(const std::size_t worker_count) {
        if (worker_count == 0U) {
            throw std::invalid_argument("ThreadPool requires at least one worker");
        }
        workers_.reserve(worker_count);
        try {
            for (std::size_t i = 0; i < worker_count; ++i) {
                workers_.emplace_back([this] { worker_loop(); });
            }
        } catch (...) {
            {
                std::lock_guard<std::mutex> lock(queue_mutex_);
                stopping_ = true;
            }
            condition_.notify_all();
            for (std::thread& worker : workers_) {
                if (worker.joinable()) {
                    worker.join();
                }
            }
            throw;
        }
    }

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            stopping_ = true;
        }
        condition_.notify_all();
        for (std::thread& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

    [[nodiscard]] std::size_t worker_count() const noexcept {
        return workers_.size();
    }

    template <typename Function>
    void parallel_for(const std::size_t count, Function&& function) {
        if (count == 0U) {
            return;
        }

        const std::size_t task_count = std::min(workers_.size(), count);
        std::atomic<std::size_t> next_index{0U};
        auto shared_function = std::make_shared<std::decay_t<Function>>(
            std::forward<Function>(function));

        std::vector<std::future<void>> completions;
        completions.reserve(task_count);
        for (std::size_t task = 0; task < task_count; ++task) {
            completions.emplace_back(submit([shared_function, &next_index, count] {
                while (true) {
                    const std::size_t index = next_index.fetch_add(1U, std::memory_order_relaxed);
                    if (index >= count) {
                        break;
                    }
                    (*shared_function)(index);
                }
            }));
        }

        // Drain every task before propagating an exception so that references
        // captured by worker tasks remain valid until all workers have stopped.
        std::exception_ptr first_exception;
        for (std::future<void>& completion : completions) {
            try {
                completion.get();
            } catch (...) {
                if (!first_exception) {
                    first_exception = std::current_exception();
                }
            }
        }
        if (first_exception) {
            std::rethrow_exception(first_exception);
        }
    }

private:
    template <typename Function>
    auto submit(Function&& function) -> std::future<std::invoke_result_t<Function>> {
        using ReturnType = std::invoke_result_t<Function>;
        auto task = std::make_shared<std::packaged_task<ReturnType()>>(
            std::forward<Function>(function));
        std::future<ReturnType> completion = task->get_future();
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            if (stopping_) {
                throw std::runtime_error("Cannot submit work to a stopped ThreadPool");
            }
            tasks_.emplace([task] { (*task)(); });
        }
        condition_.notify_one();
        return completion;
    }

    void worker_loop() {
        while (true) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                condition_.wait(lock, [this] { return stopping_ || !tasks_.empty(); });
                if (stopping_ && tasks_.empty()) {
                    return;
                }
                task = std::move(tasks_.front());
                tasks_.pop();
            }
            task();
        }
    }

    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex queue_mutex_;
    std::condition_variable condition_;
    bool stopping_{false};
};

}  // namespace cpso
