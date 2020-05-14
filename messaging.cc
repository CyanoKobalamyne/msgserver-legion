#include <ctime>
#include <deque>
#include <list>
#include <numeric>

#include "getopt.h"
#include "legion.h"

using namespace Legion;

enum TaskID {
    INIT_TASK,
    DISPATCH_TASK,
    PREPARE_FETCH_TASK,
    EXECUTE_FETCH_TASK,
    PREPARE_POST_TASK,
    EXECUTE_POST_TASK,
};

enum UserFieldID {
    FOLLOWED_CHANNEL_ID_1,
    FOLLOWED_CHANNEL_ID_2,
    FOLLOWED_CHANNEL_ID_3,
    FOLLOWED_CHANNEL_ID_4,
    FOLLOWED_CHANNEL_ID_5,
};

enum ChannelFieldID {
    NEXT_MSG_ID,
};

enum UserFirstUnreadFieldID {
    NEXT_UNREAD_MSG_ID_1,
    NEXT_UNREAD_MSG_ID_2,
    NEXT_UNREAD_MSG_ID_3,
    NEXT_UNREAD_MSG_ID_4,
    NEXT_UNREAD_MSG_ID_5,
};

enum MessageFieldID {
    AUTHOR_ID,
    TIMESTAMP,
    TEXT,
};

constexpr unsigned int MESSAGE_LENGTH = 256;

typedef uint16_t user_id_t;
typedef uint8_t channel_id_t;
typedef uint32_t message_id_t;

typedef struct {
    message_id_t message_id;
    user_id_t author_id;
    time_t timestamp;
    char text[MESSAGE_LENGTH];
} Message;

typedef struct {
    user_id_t user_id;
    channel_id_t watched_channel_ids[5];
} PrepareFetchData;

typedef struct {
    message_id_t next_unread_msg_ids[5];
    message_id_t next_channel_msg_ids[5];
} PrepareFetchResponse;

typedef struct {
    user_id_t user_id;
    channel_id_t watched_channel_ids[5];
    message_id_t next_unread_msg_ids[5];
    message_id_t next_channel_msg_ids[5];
} ExecuteFetchData;

typedef struct {
    bool success;
    message_id_t num_messages;
    Message messages[];
} ExecuteFetchResponse;

typedef struct {
    channel_id_t channel_id;
} PreparePostData;

typedef struct {
    message_id_t next_channel_msg_id;
} PreparePostResponse;

typedef struct {
    channel_id_t channel_id;
    message_id_t next_channel_msg_id;
    Message message;
} ExecutePostData;

typedef struct {
    bool success;
} ExecutePostResponse;

enum Action {
    POST,
    FETCH,
};

typedef struct {
    Action action;
    user_id_t user_id;
    channel_id_t channel_id;
    char message[MESSAGE_LENGTH];
} Request;

typedef struct {
    Future future;
    Request request;
} PendingRequest;

const struct option options[] = {
    {.name = "n", .has_arg = required_argument, .flag = NULL, .val = 'n'},
    {.name = "k", .has_arg = required_argument, .flag = NULL, .val = 'k'},
    {.name = "m", .has_arg = required_argument, .flag = NULL, .val = 'm'},
    {.name = "t", .has_arg = required_argument, .flag = NULL, .val = 't'},
    {.name = "r", .has_arg = required_argument, .flag = NULL, .val = 'r'},
    {0, 0, 0, 0},
};

void dispatch_task(const Task *task,
                   const std::vector<PhysicalRegion> &regions, Context ctx,
                   Runtime *runtime) {
    const InputArgs &args = Runtime::get_input_args();
    unsigned int user_count = 0;
    unsigned int channel_count = 0;
    unsigned int msg_count = 0;
    unsigned int n_requests = 0;
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
            n_requests = atoi(optarg);
            break;
        case 'r':
            request_ratio = atoi(optarg);
            break;
        case '?':
        default:
            break;
        }
    }

    if (user_count == 0 || channel_count == 0 || msg_count == 0 ||
        n_requests == 0) {
        std::cout << "Usage: " << args.argv[0]
                  << " [-n num_users] [-k num_channels] [-m num_messages] [-t "
                     "test_requests] [-r test_request_ratio]"
                  << std::endl;
        exit(EXIT_FAILURE);
    }

    /* Users array. */
    Rect<1> user_id_range(0, user_count);
    IndexSpaceT<1> user_ids = runtime->create_index_space(ctx, user_id_range);
    IndexPartition user_id_partition =
        runtime->create_equal_partition(ctx, user_ids, user_ids);
    FieldSpace user_fields = runtime->create_field_space(ctx);
    FieldAllocator allocator =
        runtime->create_field_allocator(ctx, user_fields);
    allocator.allocate_field(sizeof(channel_id_t), FOLLOWED_CHANNEL_ID_1);
    allocator.allocate_field(sizeof(channel_id_t), FOLLOWED_CHANNEL_ID_2);
    allocator.allocate_field(sizeof(channel_id_t), FOLLOWED_CHANNEL_ID_3);
    allocator.allocate_field(sizeof(channel_id_t), FOLLOWED_CHANNEL_ID_4);
    allocator.allocate_field(sizeof(channel_id_t), FOLLOWED_CHANNEL_ID_5);
    LogicalRegionT<1> users =
        runtime->create_logical_region(ctx, user_ids, user_fields);
    // Initialize array.
    RegionRequirement user_init_req(users, WRITE_DISCARD, EXCLUSIVE, users);
    user_init_req.add_field(FOLLOWED_CHANNEL_ID_1);
    user_init_req.add_field(FOLLOWED_CHANNEL_ID_2);
    user_init_req.add_field(FOLLOWED_CHANNEL_ID_3);
    user_init_req.add_field(FOLLOWED_CHANNEL_ID_4);
    user_init_req.add_field(FOLLOWED_CHANNEL_ID_5);
    InlineLauncher user_init_launcher(user_init_req);
    PhysicalRegion init_region = runtime->map_region(ctx, user_init_launcher);
    const FieldAccessor<WRITE_DISCARD, channel_id_t, 1> channel_id_1(
        init_region, FOLLOWED_CHANNEL_ID_1);
    const FieldAccessor<WRITE_DISCARD, channel_id_t, 1> channel_id_2(
        init_region, FOLLOWED_CHANNEL_ID_2);
    const FieldAccessor<WRITE_DISCARD, channel_id_t, 1> channel_id_3(
        init_region, FOLLOWED_CHANNEL_ID_3);
    const FieldAccessor<WRITE_DISCARD, channel_id_t, 1> channel_id_4(
        init_region, FOLLOWED_CHANNEL_ID_4);
    const FieldAccessor<WRITE_DISCARD, channel_id_t, 1> channel_id_5(
        init_region, FOLLOWED_CHANNEL_ID_5);
    std::vector<int> all_channel_ids(channel_count);
    std::iota(all_channel_ids.begin(), all_channel_ids.end(), 0);
    for (PointInRectIterator<1> iter(user_id_range); iter(); iter++) {
        std::random_shuffle(all_channel_ids.begin(), all_channel_ids.end());
        channel_id_1[*iter] = all_channel_ids[0];
        channel_id_2[*iter] = all_channel_ids[1];
        channel_id_3[*iter] = all_channel_ids[2];
        channel_id_4[*iter] = all_channel_ids[3];
        channel_id_5[*iter] = all_channel_ids[4];
    }
    // Leave this region mapped.

    /* Next unread array. */
    FieldSpace next_unread_fields = runtime->create_field_space(ctx);
    allocator = runtime->create_field_allocator(ctx, next_unread_fields);
    allocator.allocate_field(sizeof(message_id_t), NEXT_UNREAD_MSG_ID_1);
    allocator.allocate_field(sizeof(message_id_t), NEXT_UNREAD_MSG_ID_2);
    allocator.allocate_field(sizeof(message_id_t), NEXT_UNREAD_MSG_ID_3);
    allocator.allocate_field(sizeof(message_id_t), NEXT_UNREAD_MSG_ID_4);
    allocator.allocate_field(sizeof(message_id_t), NEXT_UNREAD_MSG_ID_5);
    LogicalRegionT<1> next_unreads =
        runtime->create_logical_region(ctx, user_ids, next_unread_fields);
    LogicalPartition next_unread_partition =
        runtime->get_logical_partition(next_unreads, user_id_partition);
    // Initialize array.
    RegionRequirement next_unread_init_req(next_unreads, WRITE_DISCARD,
                                           EXCLUSIVE, next_unreads);
    next_unread_init_req.add_field(NEXT_UNREAD_MSG_ID_1);
    next_unread_init_req.add_field(NEXT_UNREAD_MSG_ID_2);
    next_unread_init_req.add_field(NEXT_UNREAD_MSG_ID_3);
    next_unread_init_req.add_field(NEXT_UNREAD_MSG_ID_4);
    next_unread_init_req.add_field(NEXT_UNREAD_MSG_ID_5);
    InlineLauncher next_unread_init_launcher(next_unread_init_req);
    init_region = runtime->map_region(ctx, next_unread_init_launcher);
    const FieldAccessor<WRITE_DISCARD, message_id_t, 1> next_unread_1(
        init_region, NEXT_UNREAD_MSG_ID_1);
    const FieldAccessor<WRITE_DISCARD, message_id_t, 1> next_unread_2(
        init_region, NEXT_UNREAD_MSG_ID_2);
    const FieldAccessor<WRITE_DISCARD, message_id_t, 1> next_unread_3(
        init_region, NEXT_UNREAD_MSG_ID_3);
    const FieldAccessor<WRITE_DISCARD, message_id_t, 1> next_unread_4(
        init_region, NEXT_UNREAD_MSG_ID_4);
    const FieldAccessor<WRITE_DISCARD, message_id_t, 1> next_unread_5(
        init_region, NEXT_UNREAD_MSG_ID_5);
    for (PointInRectIterator<1> iter(user_id_range); iter(); iter++) {
        next_unread_1[*iter] = 0;
        next_unread_2[*iter] = 0;
        next_unread_3[*iter] = 0;
        next_unread_4[*iter] = 0;
        next_unread_5[*iter] = 0;
    }
    runtime->unmap_region(ctx, init_region);

    /* Channels array. */
    Rect<1> channel_id_range(0, channel_count);
    IndexSpaceT<1> channel_ids =
        runtime->create_index_space(ctx, channel_id_range);
    IndexPartition channel_id_partition =
        runtime->create_equal_partition(ctx, channel_ids, channel_ids);
    FieldSpace channel_fields = runtime->create_field_space(ctx);
    allocator = runtime->create_field_allocator(ctx, channel_fields);
    allocator.allocate_field(sizeof(message_id_t), NEXT_MSG_ID);
    LogicalRegionT<1> channels =
        runtime->create_logical_region(ctx, channel_ids, channel_fields);
    LogicalPartition channel_partition =
        runtime->get_logical_partition(channels, channel_id_partition);
    // Initialize array.
    RegionRequirement init_req(channels, WRITE_DISCARD, EXCLUSIVE, channels);
    init_req.add_field(NEXT_MSG_ID);
    InlineLauncher init_launcher(init_req);
    init_region = runtime->map_region(ctx, init_launcher);
    const FieldAccessor<WRITE_DISCARD, message_id_t, 1> next_msg(init_region,
                                                                 NEXT_MSG_ID);
    for (PointInRectIterator<1> iter(channel_id_range); iter(); iter++) {
        next_msg[*iter] = 0;
    }
    runtime->unmap_region(ctx, init_region);

    /* Messages array. */
    Rect<2> msg_id_range(Point<2>(0, 0), Point<2>(channel_count, msg_count));
    IndexSpaceT<2> msg_ids = runtime->create_index_space(ctx, msg_id_range);
    IndexPartition msg_id_partition =
        runtime->create_equal_partition(ctx, msg_ids, msg_ids);
    FieldSpace msg_fields = runtime->create_field_space(ctx);
    allocator = runtime->create_field_allocator(ctx, msg_fields);
    allocator.allocate_field(sizeof(user_id_t), AUTHOR_ID);
    allocator.allocate_field(sizeof(user_id_t), TIMESTAMP);
    allocator.allocate_field(MESSAGE_LENGTH * sizeof(char), TEXT);
    LogicalRegionT<2> messages =
        runtime->create_logical_region(ctx, msg_ids, msg_fields);
    LogicalPartition message_partition =
        runtime->get_logical_partition(messages, msg_id_partition);

    /* Generate random requests. */
    std::deque<Request> requests(n_requests);
    // TODO: actually generate.

    /* Execute requests. */
    std::list<PendingRequest> pending_reqs;
    std::vector<Future> executing_reqs;
    long time = 0;
    while (requests.size() != 0 && pending_reqs.size() != 0) {
        for (auto it = pending_reqs.begin(); it != pending_reqs.end(); it++) {
            auto &req = *it;
            if (req.future.is_ready()) {
                switch (req.request.action) {
                case FETCH: {
                    PrepareFetchResponse response =
                        req.future.get_result<PrepareFetchResponse>();
                    pending_reqs.erase(it);
                    ExecuteFetchData data = {.user_id = req.request.user_id};
                    data.watched_channel_ids[0] =
                        channel_id_1[req.request.user_id];
                    data.watched_channel_ids[1] =
                        channel_id_2[req.request.user_id];
                    data.watched_channel_ids[2] =
                        channel_id_3[req.request.user_id];
                    data.watched_channel_ids[3] =
                        channel_id_4[req.request.user_id];
                    data.watched_channel_ids[4] =
                        channel_id_5[req.request.user_id];
                    memcpy(data.next_channel_msg_ids,
                           response.next_channel_msg_ids,
                           sizeof response.next_channel_msg_ids);
                    memcpy(data.next_unread_msg_ids,
                           response.next_unread_msg_ids,
                           sizeof response.next_unread_msg_ids);
                    TaskLauncher launcher(
                        EXECUTE_FETCH_TASK,
                        TaskArgument(&data, sizeof(ExecuteFetchData)));
                    launcher.add_region_requirement(RegionRequirement(
                        runtime->get_logical_subregion_by_color(
                            next_unread_partition, req.request.user_id),
                        READ_WRITE, EXCLUSIVE, next_unreads));
                    launcher.add_field(0, NEXT_UNREAD_MSG_ID_1);
                    launcher.add_field(1, NEXT_UNREAD_MSG_ID_2);
                    launcher.add_field(2, NEXT_UNREAD_MSG_ID_3);
                    launcher.add_field(3, NEXT_UNREAD_MSG_ID_4);
                    launcher.add_field(4, NEXT_UNREAD_MSG_ID_5);
                    for (message_id_t i = data.next_unread_msg_ids[0];
                         i < data.next_channel_msg_ids[0]; i++) {
                        launcher.add_region_requirement(RegionRequirement(
                            runtime->get_logical_subregion_by_color(
                                message_partition, i),
                            READ_ONLY, EXCLUSIVE, messages));
                        launcher.add_field(0, NEXT_MSG_ID);
                    }
                    for (message_id_t i = data.next_unread_msg_ids[1];
                         i < data.next_channel_msg_ids[1]; i++) {
                        launcher.add_region_requirement(RegionRequirement(
                            runtime->get_logical_subregion_by_color(
                                message_partition, msg_count + i),
                            READ_ONLY, EXCLUSIVE, messages));
                        launcher.add_field(0, NEXT_MSG_ID);
                    }
                    for (message_id_t i = data.next_unread_msg_ids[2];
                         i < data.next_channel_msg_ids[2]; i++) {
                        launcher.add_region_requirement(RegionRequirement(
                            runtime->get_logical_subregion_by_color(
                                message_partition, msg_count * 2 + i),
                            READ_ONLY, EXCLUSIVE, messages));
                        launcher.add_field(0, NEXT_MSG_ID);
                    }
                    for (message_id_t i = data.next_unread_msg_ids[3];
                         i < data.next_channel_msg_ids[3]; i++) {
                        launcher.add_region_requirement(RegionRequirement(
                            runtime->get_logical_subregion_by_color(
                                message_partition, msg_count * 3 + i),
                            READ_ONLY, EXCLUSIVE, messages));
                        launcher.add_field(0, NEXT_MSG_ID);
                    }
                    for (message_id_t i = data.next_unread_msg_ids[4];
                         i < data.next_channel_msg_ids[4]; i++) {
                        launcher.add_region_requirement(RegionRequirement(
                            runtime->get_logical_subregion_by_color(
                                message_partition, msg_count * 4 + i),
                            READ_ONLY, EXCLUSIVE, messages));
                        launcher.add_field(0, NEXT_MSG_ID);
                    }
                    executing_reqs.push_back(
                        runtime->execute_task(ctx, launcher));
                    break;
                }
                case POST: {
                    PreparePostResponse response =
                        req.future.get_result<PreparePostResponse>();
                    pending_reqs.erase(it);
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
                    TaskLauncher launcher(
                        EXECUTE_POST_TASK,
                        TaskArgument(&data, sizeof(PrepareFetchData)));
                    launcher.add_region_requirement(RegionRequirement(
                        runtime->get_logical_subregion_by_color(
                            channel_partition, req.request.channel_id),
                        READ_WRITE, EXCLUSIVE, channels));
                    launcher.add_field(0, NEXT_MSG_ID);
                    launcher.add_region_requirement(RegionRequirement(
                        runtime->get_logical_subregion_by_color(
                            message_partition,
                            req.request.channel_id * msg_count +
                                response.next_channel_msg_id),
                        WRITE_DISCARD, EXCLUSIVE, messages));
                    launcher.add_field(0, AUTHOR_ID);
                    launcher.add_field(0, TIMESTAMP);
                    launcher.add_field(0, TEXT);
                    executing_reqs.push_back(
                        runtime->execute_task(ctx, launcher));
                    break;
                }
                }
                break;
            }
        }

        Request request = requests.back();
        requests.pop_back();
        switch (request.action) {
        case FETCH: {
            PrepareFetchData data = {.user_id = request.user_id};
            data.watched_channel_ids[0] = channel_id_1[request.user_id];
            data.watched_channel_ids[1] = channel_id_2[request.user_id];
            data.watched_channel_ids[2] = channel_id_3[request.user_id];
            data.watched_channel_ids[3] = channel_id_4[request.user_id];
            data.watched_channel_ids[4] = channel_id_5[request.user_id];
            TaskLauncher launcher(
                PREPARE_FETCH_TASK,
                TaskArgument(&data, sizeof(PrepareFetchData)));
            launcher.add_region_requirement(
                RegionRequirement(runtime->get_logical_subregion_by_color(
                                      next_unread_partition, request.user_id),
                                  READ_ONLY, EXCLUSIVE, next_unreads));
            launcher.add_field(0, NEXT_UNREAD_MSG_ID_1);
            launcher.add_field(1, NEXT_UNREAD_MSG_ID_2);
            launcher.add_field(2, NEXT_UNREAD_MSG_ID_3);
            launcher.add_field(3, NEXT_UNREAD_MSG_ID_4);
            launcher.add_field(4, NEXT_UNREAD_MSG_ID_5);
            launcher.add_region_requirement(RegionRequirement(
                runtime->get_logical_subregion_by_color(
                    channel_partition, channel_id_1[request.user_id]),
                READ_ONLY, EXCLUSIVE, channels));
            launcher.add_field(0, NEXT_MSG_ID);
            launcher.add_region_requirement(RegionRequirement(
                runtime->get_logical_subregion_by_color(
                    channel_partition, channel_id_2[request.user_id]),
                READ_ONLY, EXCLUSIVE, channels));
            launcher.add_field(0, NEXT_MSG_ID);
            launcher.add_region_requirement(RegionRequirement(
                runtime->get_logical_subregion_by_color(
                    channel_partition, channel_id_3[request.user_id]),
                READ_ONLY, EXCLUSIVE, channels));
            launcher.add_field(0, NEXT_MSG_ID);
            launcher.add_region_requirement(RegionRequirement(
                runtime->get_logical_subregion_by_color(
                    channel_partition, channel_id_4[request.user_id]),
                READ_ONLY, EXCLUSIVE, channels));
            launcher.add_field(0, NEXT_MSG_ID);
            launcher.add_region_requirement(RegionRequirement(
                runtime->get_logical_subregion_by_color(
                    channel_partition, channel_id_5[request.user_id]),
                READ_ONLY, EXCLUSIVE, channels));
            launcher.add_field(0, NEXT_MSG_ID);
            pending_reqs.push_back(
                {.future = runtime->execute_task(ctx, launcher),
                 .request = request});
            break;
        }

        case POST: {
            PreparePostData data = {.channel_id = request.channel_id};
            TaskLauncher launcher(
                PREPARE_POST_TASK,
                TaskArgument(&data, sizeof(PreparePostData)));
            launcher.add_region_requirement(
                RegionRequirement(runtime->get_logical_subregion_by_color(
                                      channel_partition, request.channel_id),
                                  READ_ONLY, EXCLUSIVE, channels));
            launcher.add_field(0, NEXT_MSG_ID);
            pending_reqs.push_back(
                {.future = runtime->execute_task(ctx, launcher),
                 .request = request});
            break;
        }
        }
    }
}

PrepareFetchResponse prepare_fetch_task(
    const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx,
    Runtime *runtime) {
    PrepareFetchData *data = (PrepareFetchData *)task->args;
    PrepareFetchResponse response;
    // TOFO: perform task.
    return response;
}

ExecuteFetchResponse execute_fetch_task(
    const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx,
    Runtime *runtime) {
    ExecuteFetchData *data = (ExecuteFetchData *)task->args;
    ExecuteFetchResponse response;
    // TOFO: perform task.
    return response;
}

PreparePostResponse prepare_post_task(
    const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx,
    Runtime *runtime) {
    PreparePostData *data = (PreparePostData *)task->args;
    PreparePostResponse response;
    // TOFO: perform task.
    return response;
}

ExecutePostResponse execute_post_task(
    const Task *task, const std::vector<PhysicalRegion> &regions, Context ctx,
    Runtime *runtime) {
    ExecutePostData *data = (ExecutePostData *)task->args;
    ExecutePostResponse response;
    // TOFO: perform task.
    return response;
}

int main(int argc, char **argv) {
    Runtime::set_top_level_task_id(DISPATCH_TASK);

    {
        TaskVariantRegistrar registrar(DISPATCH_TASK, "dispatch");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<dispatch_task>(registrar,
                                                         "dispatch");
    }

    {
        TaskVariantRegistrar registrar(PREPARE_FETCH_TASK, "prepare_fetch");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<PrepareFetchResponse,
                                          prepare_fetch_task>(registrar,
                                                              "prepare_fetch");
    }

    {
        TaskVariantRegistrar registrar(EXECUTE_FETCH_TASK, "execute_fetch");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<ExecuteFetchResponse,
                                          execute_fetch_task>(registrar,
                                                              "execute_fetch");
    }

    {
        TaskVariantRegistrar registrar(PREPARE_POST_TASK, "prepare_post");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<PreparePostResponse,
                                          prepare_post_task>(registrar,
                                                             "prepare_post");
    }

    {
        TaskVariantRegistrar registrar(EXECUTE_POST_TASK, "execute_post");
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
        Runtime::preregister_task_variant<ExecutePostResponse,
                                          execute_post_task>(registrar,
                                                             "execute_post");
    }

    return Runtime::start(argc, argv);
}
