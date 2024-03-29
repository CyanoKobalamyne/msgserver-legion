#include <chrono>
#include <cstring>
#include <ctime>
#include <deque>
#include <functional>
#include <iomanip>
#include <iostream>
#include <list>
#include <numeric>
#include <random>
#include <string>

#include "getopt.h"
#include "legion.h"

enum TaskID {
    INIT_TASK,
    DISPATCH_TASK,
    PREPARE_FETCH_TASK,
    EXECUTE_FETCH_TASK,
    PREPARE_POST_TASK,
    EXECUTE_POST_TASK,
};

enum UserFieldID { FOLLOWED_CHANNEL_IDS };

enum ChannelFieldID {
    NEXT_MSG_ID,
};

enum UserFirstUnreadFieldID { NEXT_UNREAD_MSG_IDS };

enum MessageFieldID {
    AUTHOR_ID,
    TIMESTAMP,
    TEXT,
};

constexpr unsigned int CHANNELS_PER_USER = 4;
constexpr unsigned int MESSAGE_LENGTH = 256;
constexpr unsigned int MAX_RETURNED_MESSAGES = 20;
constexpr char msg_template[] = "This is a message from user %d on channel %d";

typedef uint16_t user_id_t;
typedef uint8_t channel_id_t;
typedef uint32_t message_id_t;

class MessageText {
private:
    char buffer[MESSAGE_LENGTH];

public:
    MessageText() { strcpy(buffer, ""); }
    MessageText(std::string str) { strcpy(buffer, str.c_str()); }
    MessageText(char *str) { strcpy(buffer, str); }

    operator void *() { return buffer; }
    operator char *() { return buffer; }
    char &operator[](size_t n) { return buffer[n]; }
    MessageText &operator=(MessageText text) {
        memcpy(buffer, text, sizeof buffer);
        return *this;
    }
};

template <typename T>
class PerUserChannel {
private:
    T array[CHANNELS_PER_USER];

public:
    PerUserChannel() {
        for (size_t i = 0; i < CHANNELS_PER_USER; i++) array[i] = T();
    }
    PerUserChannel(T src[]) { memcpy(array, src, sizeof array); }

    operator void *() { return array; }
    T &operator[](size_t n) { return array[n]; }
};

struct Message {
    message_id_t message_id;
    user_id_t author_id;
    time_t timestamp;
    MessageText text;
};

class MessageList {
private:
    Message msgs[MAX_RETURNED_MESSAGES];

public:
    Message &operator[](size_t n) { return msgs[n]; }
};

struct PrepareFetchData {
    user_id_t user_id;
    PerUserChannel<channel_id_t> watched_channel_ids;
};

struct PrepareFetchResponse {
    PerUserChannel<message_id_t> next_unread_msg_ids;
    PerUserChannel<message_id_t> next_channel_msg_ids;
};

struct ExecuteFetchData {
    user_id_t user_id;
    PerUserChannel<channel_id_t> watched_channel_ids;
    PerUserChannel<message_id_t> next_unread_msg_ids;
    PerUserChannel<message_id_t> next_channel_msg_ids;
};

struct ExecuteFetchResponse {
    bool success;
    message_id_t num_messages;
    MessageList messages;
};

struct PreparePostData {
    channel_id_t channel_id;
};

struct PreparePostResponse {
    message_id_t next_channel_msg_id;
};

struct ExecutePostData {
    channel_id_t channel_id;
    message_id_t next_channel_msg_id;
    Message message;
};

struct ExecutePostResponse {
    bool success;
};

enum Action {
    POST,
    FETCH,
};

struct Request {
    Action action;
    user_id_t user_id;
    channel_id_t channel_id;
    MessageText message;
};

struct PendingRequest {
    Legion::Future future;
    Request request;
};

const struct option options[] = {
    {.name = "n", .has_arg = required_argument, .flag = NULL, .val = 'n'},
    {.name = "k", .has_arg = required_argument, .flag = NULL, .val = 'k'},
    {.name = "m", .has_arg = required_argument, .flag = NULL, .val = 'm'},
    {.name = "t", .has_arg = required_argument, .flag = NULL, .val = 't'},
    {.name = "r", .has_arg = required_argument, .flag = NULL, .val = 'r'},
    {0, 0, 0, 0},
};

unsigned long long fetch_time;
unsigned long long fetch_count;
unsigned long long fetch_message_count;
unsigned long long post_time;
unsigned long long post_count;

void dispatch_task(const Legion::Task *task,
                   const std::vector<Legion::PhysicalRegion> &regions,
                   Legion::Context ctx, Legion::Runtime *runtime) {
    const Legion::InputArgs &args = Legion::Runtime::get_input_args();
    user_id_t user_count = 0;
    channel_id_t channel_count = 0;
    message_id_t msg_count = 0;
    unsigned long n_requests = 0;
    unsigned int request_ratio = 1;

    int opt;
    opterr = 0;
    while ((opt = getopt_long_only(args.argc, args.argv, "", options, NULL)) !=
           -1) {
        switch (opt) {
        case 'n':
            user_count = atoi(optarg);
            break;
        case 'k':
            channel_count = atoi(optarg);
            break;
        case 'm':
            msg_count = atoi(optarg);
            break;
        case 't':
            n_requests = atol(optarg);
            break;
        case 'r':
            request_ratio = atoi(optarg);
            break;
        case '?':
        default:
            break;
        }
    }

    // Check that (nonzero) arguments are given.
    if (user_count == 0 || channel_count == 0 || msg_count == 0 ||
        n_requests == 0 || request_ratio == 0) {
        std::cerr << "Usage: " << args.argv[0]
                  << " [-n num_users] [-k num_channels] [-m num_messages] [-t "
                     "test_requests] [-r test_request_ratio]"
                  << std::endl;
        exit(EXIT_FAILURE);
    }

    // Check that we have enough channels to choose from.
    if (channel_count < CHANNELS_PER_USER) {
        std::cerr << "You must specify at least " << CHANNELS_PER_USER
                  << " channels" << std::endl;
        exit(EXIT_FAILURE);
    }

    unsigned long n_post_requests = n_requests / (request_ratio + 1);
    unsigned long n_fetch_requests = n_post_requests * request_ratio;

    // Check that we will have at lest one request.
    if (n_post_requests == 0) {
        std::cerr
            << "The number of requests is too low for the chosen ratio.\n"
            << "Please increase the number of requests or decrease the ratio."
            << std::endl;
        exit(EXIT_FAILURE);
    }

    std::default_random_engine rng;

    /* Users array. */
    Legion::Rect<1> user_id_range(0, user_count);
    Legion::IndexSpaceT<1> user_ids =
        runtime->create_index_space(ctx, user_id_range);
    Legion::IndexPartition user_id_partition =
        runtime->create_equal_partition(ctx, user_ids, user_ids);
    Legion::FieldSpace user_fields = runtime->create_field_space(ctx);
    Legion::FieldAllocator allocator =
        runtime->create_field_allocator(ctx, user_fields);
    allocator.allocate_field(CHANNELS_PER_USER * sizeof(channel_id_t),
                             FOLLOWED_CHANNEL_IDS);
    Legion::LogicalRegionT<1> users =
        runtime->create_logical_region(ctx, user_ids, user_fields);
    // Initialize array.
    Legion::RegionRequirement user_init_req(users, WRITE_DISCARD, EXCLUSIVE,
                                            users);
    user_init_req.add_field(FOLLOWED_CHANNEL_IDS);
    Legion::InlineLauncher user_init_launcher(user_init_req);
    Legion::PhysicalRegion init_region =
        runtime->map_region(ctx, user_init_launcher);
    const Legion::FieldAccessor<WRITE_DISCARD, PerUserChannel<channel_id_t>, 1>
        channel_id_mem(init_region, FOLLOWED_CHANNEL_IDS);
    std::vector<channel_id_t> all_channel_ids(channel_count);
    std::iota(all_channel_ids.begin(), all_channel_ids.end(), 0);
    for (Legion::PointInRectIterator<1> iter(user_id_range); iter(); iter++) {
        std::shuffle(all_channel_ids.begin(), all_channel_ids.end(), rng);
        channel_id_mem[*iter] =
            PerUserChannel<channel_id_t>(&all_channel_ids[0]);
    }
    // Leave this region mapped.

    /* Next unread array. */
    Legion::FieldSpace next_unread_fields = runtime->create_field_space(ctx);
    allocator = runtime->create_field_allocator(ctx, next_unread_fields);
    allocator.allocate_field(CHANNELS_PER_USER * sizeof(message_id_t),
                             NEXT_UNREAD_MSG_IDS);
    Legion::LogicalRegionT<1> next_unreads =
        runtime->create_logical_region(ctx, user_ids, next_unread_fields);
    Legion::LogicalPartition next_unread_partition =
        runtime->get_logical_partition(next_unreads, user_id_partition);
    // Initialize array.
    Legion::RegionRequirement next_unread_init_req(next_unreads, WRITE_DISCARD,
                                                   EXCLUSIVE, next_unreads);
    next_unread_init_req.add_field(NEXT_UNREAD_MSG_IDS);
    Legion::InlineLauncher next_unread_init_launcher(next_unread_init_req);
    init_region = runtime->map_region(ctx, next_unread_init_launcher);
    const Legion::FieldAccessor<WRITE_DISCARD, PerUserChannel<message_id_t>, 1>
        next_unread_mem(init_region, NEXT_UNREAD_MSG_IDS);
    for (Legion::PointInRectIterator<1> iter(user_id_range); iter(); iter++) {
        next_unread_mem[*iter] = PerUserChannel<message_id_t>();
    }
    runtime->unmap_region(ctx, init_region);

    /* Channels array. */
    Legion::Rect<1> channel_id_range(0, channel_count);
    Legion::IndexSpaceT<1> channel_ids =
        runtime->create_index_space(ctx, channel_id_range);
    Legion::IndexPartition channel_id_partition =
        runtime->create_equal_partition(ctx, channel_ids, channel_ids);
    Legion::FieldSpace channel_fields = runtime->create_field_space(ctx);
    allocator = runtime->create_field_allocator(ctx, channel_fields);
    allocator.allocate_field(sizeof(message_id_t), NEXT_MSG_ID);
    Legion::LogicalRegionT<1> channels =
        runtime->create_logical_region(ctx, channel_ids, channel_fields);
    Legion::LogicalPartition channel_partition =
        runtime->get_logical_partition(channels, channel_id_partition);
    // Initialize array.
    Legion::RegionRequirement init_req(channels, WRITE_DISCARD, EXCLUSIVE,
                                       channels);
    init_req.add_field(NEXT_MSG_ID);
    Legion::InlineLauncher init_launcher(init_req);
    init_region = runtime->map_region(ctx, init_launcher);
    const Legion::FieldAccessor<WRITE_DISCARD, message_id_t, 1> next_msg(
        init_region, NEXT_MSG_ID);
    for (Legion::PointInRectIterator<1> iter(channel_id_range); iter();
         iter++) {
        next_msg[*iter] = 0;
    }
    runtime->unmap_region(ctx, init_region);

    /* Messages array. */
    Legion::Rect<2> msg_id_range(Legion::Point<2>(0, 0),
                                 Legion::Point<2>(channel_count, msg_count));
    Legion::IndexSpaceT<2> msg_ids =
        runtime->create_index_space(ctx, msg_id_range);
    Legion::IndexPartition msg_id_partition =
        runtime->create_equal_partition(ctx, msg_ids, msg_ids);
    Legion::FieldSpace msg_fields = runtime->create_field_space(ctx);
    allocator = runtime->create_field_allocator(ctx, msg_fields);
    allocator.allocate_field(sizeof(user_id_t), AUTHOR_ID);
    allocator.allocate_field(sizeof(time_t), TIMESTAMP);
    allocator.allocate_field(sizeof(MessageText), TEXT);
    Legion::LogicalRegionT<2> messages =
        runtime->create_logical_region(ctx, msg_ids, msg_fields);
    Legion::LogicalPartition message_partition =
        runtime->get_logical_partition(messages, msg_id_partition);

    /* Generate random requests. */
    std::deque<Request> requests;
    auto random_user_id = std::bind(
        std::uniform_int_distribution(user_id_t(0), user_id_t(user_count - 1)),
        rng);
    for (unsigned long i = 0; i < n_fetch_requests; i++) {
        requests.push_back({.action = FETCH, .user_id = random_user_id()});
    }
    auto random_watched_ix = std::bind(
        std::uniform_int_distribution(0U, CHANNELS_PER_USER - 1), rng);
    for (unsigned long i = 0; i < n_post_requests; i++) {
        Request req = {.action = POST, .user_id = random_user_id()};
        req.channel_id =
            ((PerUserChannel<channel_id_t>)
                 channel_id_mem[req.user_id])[random_watched_ix()];
        snprintf(req.message, MESSAGE_LENGTH, msg_template, req.user_id,
                 req.channel_id);
        requests.push_back(req);
    }
    std::shuffle(requests.begin(), requests.end(), rng);

    /* Statistics. */
    unsigned long n_failed_fetch = 0;
    unsigned long n_failed_post = 0;

    /* Execute requests. */
    std::list<PendingRequest> pending_reqs;
    std::vector<PendingRequest> executing_reqs;
    auto start = std::chrono::high_resolution_clock::now();
    time_t time = 0;
    while (requests.size() != 0 || pending_reqs.size() != 0) {
        for (auto it = pending_reqs.begin(); it != pending_reqs.end(); it++) {
            auto &req = *it;
            if (req.future.is_ready()) {
                switch (req.request.action) {
                case FETCH: {
                    PrepareFetchResponse response =
                        req.future.get_result<PrepareFetchResponse>();
                    ExecuteFetchData data = {.user_id = req.request.user_id};
                    memcpy(data.watched_channel_ids,
                           (PerUserChannel<channel_id_t>)
                               channel_id_mem[req.request.user_id],
                           sizeof data.watched_channel_ids);
                    memcpy(data.next_channel_msg_ids,
                           response.next_channel_msg_ids,
                           sizeof data.next_channel_msg_ids);
                    memcpy(data.next_unread_msg_ids,
                           response.next_unread_msg_ids,
                           sizeof data.next_unread_msg_ids);
                    Legion::TaskLauncher launcher(
                        EXECUTE_FETCH_TASK,
                        Legion::TaskArgument(&data, sizeof(ExecuteFetchData)));
                    launcher.add_region_requirement(Legion::RegionRequirement(
                        runtime->get_logical_subregion_by_color(
                            next_unread_partition, req.request.user_id),
                        READ_WRITE, EXCLUSIVE, next_unreads));
                    launcher.add_field(0, NEXT_UNREAD_MSG_IDS);
                    unsigned long reqid = 1;
                    for (unsigned int i = 0; i < CHANNELS_PER_USER; i++) {
                        for (message_id_t j = data.next_unread_msg_ids[i];
                             j < data.next_channel_msg_ids[i]; j++, reqid++) {
                            launcher.add_region_requirement(
                                Legion::RegionRequirement(
                                    runtime->get_logical_subregion_by_color(
                                        message_partition,
                                        Legion::Point<2>(
                                            data.watched_channel_ids[i], j)),
                                    READ_ONLY, EXCLUSIVE, messages));
                            launcher.add_field(reqid, NEXT_MSG_ID);
                        }
                    }
                    executing_reqs.push_back(
                        {.future = runtime->execute_task(ctx, launcher),
                         .request = req.request});
                    break;
                }
                case POST: {
                    PreparePostResponse response =
                        req.future.get_result<PreparePostResponse>();
                    Message msg = {
                        .message_id = response.next_channel_msg_id,
                        .author_id = req.request.user_id,
                        .timestamp = time,
                    };
                    memcpy(msg.text, req.request.message, sizeof msg.text);
                    time++;
                    ExecutePostData data = {
                        .channel_id = req.request.channel_id,
                        .next_channel_msg_id = response.next_channel_msg_id,
                        .message = msg};
                    Legion::TaskLauncher launcher(
                        EXECUTE_POST_TASK,
                        Legion::TaskArgument(&data, sizeof(ExecutePostData)));
                    launcher.add_region_requirement(Legion::RegionRequirement(
                        runtime->get_logical_subregion_by_color(
                            channel_partition, req.request.channel_id),
                        READ_WRITE, EXCLUSIVE, channels));
                    launcher.add_field(0, NEXT_MSG_ID);
                    launcher.add_region_requirement(Legion::RegionRequirement(
                        runtime->get_logical_subregion_by_color(
                            message_partition,
                            Legion::Point<2>(req.request.channel_id,
                                             response.next_channel_msg_id)),
                        WRITE_DISCARD, EXCLUSIVE, messages));
                    launcher.add_field(1, AUTHOR_ID);
                    launcher.add_field(1, TIMESTAMP);
                    launcher.add_field(1, TEXT);
                    executing_reqs.push_back(
                        {.future = runtime->execute_task(ctx, launcher),
                         .request = req.request});
                    break;
                }
                }
                pending_reqs.erase(it);
                break;
            }
        }

        if (requests.size() == 0) {
            continue;
        }
        Request &request = requests.front();
        switch (request.action) {
        case FETCH: {
            PrepareFetchData data = {.user_id = request.user_id};
            PerUserChannel<channel_id_t> watched_channel_ids =
                channel_id_mem[request.user_id];
            memcpy(data.watched_channel_ids, watched_channel_ids,
                   sizeof data.watched_channel_ids);
            Legion::TaskLauncher launcher(
                PREPARE_FETCH_TASK,
                Legion::TaskArgument(&data, sizeof(PrepareFetchData)));
            launcher.add_region_requirement(Legion::RegionRequirement(
                runtime->get_logical_subregion_by_color(next_unread_partition,
                                                        request.user_id),
                READ_ONLY, EXCLUSIVE, next_unreads));
            launcher.add_field(0, NEXT_UNREAD_MSG_IDS);
            for (unsigned int i = 0; i < CHANNELS_PER_USER; i++) {
                launcher.add_region_requirement(Legion::RegionRequirement(
                    runtime->get_logical_subregion_by_color(
                        channel_partition, watched_channel_ids[i]),
                    READ_ONLY, EXCLUSIVE, channels));
                launcher.add_field(1 + i, NEXT_MSG_ID);
            }
            pending_reqs.push_back(
                {.future = runtime->execute_task(ctx, launcher),
                 .request = request});
            break;
        }

        case POST: {
            PreparePostData data = {.channel_id = request.channel_id};
            Legion::TaskLauncher launcher(
                PREPARE_POST_TASK,
                Legion::TaskArgument(&data, sizeof(PreparePostData)));
            launcher.add_region_requirement(Legion::RegionRequirement(
                runtime->get_logical_subregion_by_color(channel_partition,
                                                        request.channel_id),
                READ_ONLY, EXCLUSIVE, channels));
            launcher.add_field(0, NEXT_MSG_ID);
            pending_reqs.push_back(
                {.future = runtime->execute_task(ctx, launcher),
                 .request = request});
            break;
        }
        }
        requests.pop_front();
        time++;
    }
    // Wait for all tasks to complete.
    for (auto req : executing_reqs) {
        switch (req.request.action) {
        case FETCH: {
            auto response = req.future.get_result<ExecuteFetchResponse>();
            if (!response.success) {
                n_failed_fetch++;
            }
            break;
        }

        case POST: {
            auto response = req.future.get_result<ExecutePostResponse>();
            if (!response.success) {
                n_failed_post++;
            }
            break;
        }
        }
    }

    auto stop = std::chrono::high_resolution_clock::now();

    auto duration =
        std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
    std::cout << "Time: " << duration.count() << " ns" << std::endl;

    std::cout << std::fixed << std::setprecision(0);
    std::cout << "Fetch: " << fetch_time / fetch_count << " ns average, "
              << n_failed_fetch << "/" << n_fetch_requests << " failed, "
              << fetch_message_count << " messages" << std::endl;
    std::cout << "Post: " << post_time / post_count << " ns average, "
              << n_failed_post << "/" << n_post_requests << " failed"
              << std::endl;

    return;
}

PrepareFetchResponse prepare_fetch_task(
    const Legion::Task *task,
    const std::vector<Legion::PhysicalRegion> &regions, Legion::Context ctx,
    Legion::Runtime *runtime) {
    auto start = std::chrono::high_resolution_clock::now();
    PrepareFetchData *data = (PrepareFetchData *)task->args;
    PrepareFetchResponse response;
    const Legion::FieldAccessor<READ_ONLY, PerUserChannel<message_id_t>, 1>
        next_unread(regions[0], NEXT_UNREAD_MSG_IDS);
    memcpy(response.next_unread_msg_ids,
           ((PerUserChannel<message_id_t>)next_unread[data->user_id]),
           sizeof response.next_unread_msg_ids);
    for (unsigned int i = 0; i < CHANNELS_PER_USER; i++) {
        const Legion::FieldAccessor<READ_ONLY, message_id_t, 1> next_msg(
            regions[1 + i], NEXT_MSG_ID);
        response.next_channel_msg_ids[i] =
            next_msg[data->watched_channel_ids[i]];
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    std::cerr << "[FETCH PREPARE] took " << duration.count() << " ns, user "
              << data->user_id << std::endl;
    fetch_time += duration.count();
    fetch_count++;
    return response;
}

ExecuteFetchResponse execute_fetch_task(
    const Legion::Task *task,
    const std::vector<Legion::PhysicalRegion> &regions, Legion::Context ctx,
    Legion::Runtime *runtime) {
    auto start = std::chrono::high_resolution_clock::now();
    ExecuteFetchResponse response;
    response.success = true;
    ExecuteFetchData *data = (ExecuteFetchData *)task->args;
    const Legion::FieldAccessor<READ_WRITE, PerUserChannel<message_id_t>, 1>
        next_unread(regions[0], NEXT_UNREAD_MSG_IDS);
    PerUserChannel<message_id_t> user_next_unread = next_unread[data->user_id];
    for (unsigned int i = 0; i < CHANNELS_PER_USER; i++) {
        if (data->next_unread_msg_ids[i] != user_next_unread[i]) {
            response.success = false;
            break;
        }
    }
    if (response.success) {
        unsigned long long index = 1;
        for (unsigned int i = 0; i < CHANNELS_PER_USER; i++) {
            message_id_t max_msg_id =
                std::min(data->next_channel_msg_ids[i],
                         data->next_unread_msg_ids[i] + MAX_RETURNED_MESSAGES);
            for (message_id_t j = data->next_unread_msg_ids[i]; j < max_msg_id;
                 j++, index++) {
                Legion::FieldAccessor<READ_ONLY, user_id_t, 1> author(
                    regions[index], AUTHOR_ID);
                Legion::FieldAccessor<READ_ONLY, time_t, 1> timestamp(
                    regions[index], TIMESTAMP);
                Legion::FieldAccessor<READ_ONLY, MessageText, 1> text(
                    regions[index], TEXT);
                response.messages[index] = {.message_id = j,
                                            .author_id = author[j],
                                            .timestamp = timestamp[j],
                                            .text = text[j]};
            }
            user_next_unread[i] = max_msg_id;
        }
        next_unread[data->user_id] = user_next_unread;  // Overwrite values.
        response.num_messages = index;
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    std::cerr << "[FETCH EXECUTE] took " << duration.count() << " ns, user "
              << data->user_id << (response.success ? "" : ", failed")
              << std::endl;
    fetch_time += duration.count();
    fetch_message_count += response.num_messages;
    return response;
}

PreparePostResponse prepare_post_task(
    const Legion::Task *task,
    const std::vector<Legion::PhysicalRegion> &regions, Legion::Context ctx,
    Legion::Runtime *runtime) {
    auto start = std::chrono::high_resolution_clock::now();
    PreparePostData *data = (PreparePostData *)task->args;
    PreparePostResponse response;
    Legion::FieldAccessor<READ_ONLY, message_id_t, 1> next_msg(regions[0],
                                                               NEXT_MSG_ID);
    response.next_channel_msg_id = next_msg[data->channel_id];
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    std::cerr << "[POST PREPARE] took " << duration.count() << " ns, channel "
              << (int)data->channel_id << std::endl;
    post_time += duration.count();
    post_count++;
    return response;
}

ExecutePostResponse execute_post_task(
    const Legion::Task *task,
    const std::vector<Legion::PhysicalRegion> &regions, Legion::Context ctx,
    Legion::Runtime *runtime) {
    auto start = std::chrono::high_resolution_clock::now();
    ExecutePostResponse response;
    response.success = true;
    ExecutePostData *data = (ExecutePostData *)task->args;
    Legion::FieldAccessor<READ_WRITE, message_id_t, 1> next_msg(regions[0],
                                                                NEXT_MSG_ID);
    if (next_msg[data->channel_id] != data->next_channel_msg_id) {
        response.success = false;
    }
    if (response.success) {
        Legion::Point<2> msg_id(data->channel_id, data->next_channel_msg_id);
        Legion::FieldAccessor<WRITE_DISCARD, user_id_t, 2> author(regions[1],
                                                                  AUTHOR_ID);
        author[msg_id] = data->message.author_id;
        Legion::FieldAccessor<WRITE_DISCARD, time_t, 2> timestamp(regions[1],
                                                                  TIMESTAMP);
        timestamp[msg_id] = data->message.timestamp;
        Legion::FieldAccessor<WRITE_DISCARD, MessageText, 2> text(regions[1],
                                                                  TEXT);
        text[msg_id] = data->message.text;
        next_msg[data->channel_id] = next_msg[data->channel_id] + 1;
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    std::cerr << "[POST EXECUTE] took " << duration.count() << ", channel "
              << (int)data->channel_id << (response.success ? "" : ", failed")
              << std::endl;
    post_time += duration.count();
    return response;
}

int main(int argc, char **argv) {
    Legion::Runtime::set_top_level_task_id(DISPATCH_TASK);

    {
        Legion::TaskVariantRegistrar registrar(DISPATCH_TASK, "dispatch");
        registrar.add_constraint(
            Legion::ProcessorConstraint(Legion::Processor::LOC_PROC));
        Legion::Runtime::preregister_task_variant<dispatch_task>(registrar,
                                                                 "dispatch");
    }

    {
        Legion::TaskVariantRegistrar registrar(PREPARE_FETCH_TASK,
                                               "prepare_fetch");
        registrar.add_constraint(
            Legion::ProcessorConstraint(Legion::Processor::LOC_PROC));
        Legion::Runtime::preregister_task_variant<PrepareFetchResponse,
                                                  prepare_fetch_task>(
            registrar, "prepare_fetch");
    }

    {
        Legion::TaskVariantRegistrar registrar(EXECUTE_FETCH_TASK,
                                               "execute_fetch");
        registrar.add_constraint(
            Legion::ProcessorConstraint(Legion::Processor::LOC_PROC));
        Legion::Runtime::preregister_task_variant<ExecuteFetchResponse,
                                                  execute_fetch_task>(
            registrar, "execute_fetch");
    }

    {
        Legion::TaskVariantRegistrar registrar(PREPARE_POST_TASK,
                                               "prepare_post");
        registrar.add_constraint(
            Legion::ProcessorConstraint(Legion::Processor::LOC_PROC));
        Legion::Runtime::preregister_task_variant<PreparePostResponse,
                                                  prepare_post_task>(
            registrar, "prepare_post");
    }

    {
        Legion::TaskVariantRegistrar registrar(EXECUTE_POST_TASK,
                                               "execute_post");
        registrar.add_constraint(
            Legion::ProcessorConstraint(Legion::Processor::LOC_PROC));
        Legion::Runtime::preregister_task_variant<ExecutePostResponse,
                                                  execute_post_task>(
            registrar, "execute_post");
    }

    return Legion::Runtime::start(argc, argv);
}
