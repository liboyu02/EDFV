import numpy as np

def eventcnt(events, num_bins, height, width, pol=False):
    assert (events.shape[1] == 4)
    assert (num_bins > 0)
    assert (width > 0)
    assert (height > 0)

    voxel_grid = np.zeros((num_bins, height, width), np.float32).ravel()

    # normalize the event timestamps so that they lie between 0 and num_bins
    last_stamp = events[-1, 0]
    first_stamp = events[0, 0]
    deltaT = last_stamp - first_stamp

    if deltaT == 0:
        deltaT = 1.0
    events_copy = events.copy()
    events_copy[:, 0] = (num_bins) * (events[:, 0] - first_stamp) / deltaT
    ts = events_copy[:, 0]
    xs = events[:, 1].astype(np.int32)
    ys = events[:, 2].astype(np.int32)
    pols = events[:, 3]
    pols[pols == 0] = -1  # polarity should be +1 / -1

    tis = ts.astype(np.int32)
    # print('tis',np.max(tis),np.min(tis))
    dts = ts - tis
    
    valid_indices = tis < num_bins
    if pol:
        np.add.at(voxel_grid,xs[valid_indices] + ys[valid_indices] * width +
                  tis[valid_indices] * width * height,pols[valid_indices])
    else:    
        np.add.at(voxel_grid,xs[valid_indices] + ys[valid_indices] * width +
                  tis[valid_indices] * width * height,1)

    voxel_grid = np.reshape(voxel_grid, (num_bins, height, width))
    # print('voxel_grid',np.max(voxel_grid),np.min(voxel_grid))
    # print('last layer',np.max(voxel_grid[-1]),np.min(voxel_grid[-1]))
    voxel_grid = np.moveaxis(voxel_grid, 0, -1)
    return voxel_grid

def events_to_voxel_grid(events, num_bins, height, width):
        """
        Build a voxel grid with bilinear interpolation in the time domain from a set of events.
        :param events: a [N x 4] NumPy array containing one event per row in the form: [timestamp, x, y, polarity]
        :param num_bins: number of bins in the temporal axis of the voxel grid
        :param width, height: dimensions of the voxel grid
        """

        assert (events.shape[1] == 4)
        assert (num_bins > 0)
        assert (width > 0)
        assert (height > 0)

        voxel_grid = np.zeros((num_bins, height, width), np.float32).ravel()

        # normalize the event timestamps so that they lie between 0 and num_bins
        last_stamp = events[-1, 0]
        first_stamp = events[0, 0]
        deltaT = last_stamp - first_stamp

        if deltaT == 0:
            deltaT = 1.0

        events[:, 0] = (num_bins) * (events[:, 0] - first_stamp) / deltaT
        ts = events[:, 0]
        xs = events[:, 1].astype(np.int32)
        ys = events[:, 2].astype(np.int32)
        pols = events[:, 3]
        pols[pols == 0] = -1  # polarity should be +1 / -1

        tis = ts.astype(np.int32)
        dts = ts - tis
        vals_left = pols * (1.0 - dts)
        vals_right = pols * dts

        valid_indices = tis < num_bins
        np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width +
                  tis[valid_indices] * width * height, vals_left[valid_indices])

        valid_indices = (tis + 1) < num_bins
        np.add.at(voxel_grid, xs[valid_indices] + ys[valid_indices] * width +
                  (tis[valid_indices] + 1) * width * height, vals_right[valid_indices])

        voxel_grid = np.reshape(voxel_grid, (num_bins, height, width))
        # voxel_grid = np.moveaxis(voxel_grid, 0, -1)
        return voxel_grid


def normalize_voxelgrid( event_tensor):
        # normalize the event tensor (voxel grid) in such a way that the mean and stddev of the nonzero values
        # in the tensor are equal to (0.0, 1.0)
        mask = np.nonzero(event_tensor)
        if mask[0].size > 0:
            mean, stddev = event_tensor[mask].mean(), event_tensor[mask].std()
            if stddev > 0:
                event_tensor[mask] = (event_tensor[mask] - mean) / stddev
        return event_tensor

def voxel_grid_equal_num(events,height=360,width=480,channel=3):
          
    N = events.shape[0]
    #event_stacks = [events[:N//3,:],events[N//3:N*2//3,:],events[N*2//3:,:]]
    grids = np.zeros((height,width,channel))
    for i in range(channel):
        evs = np.array(events[i*N//channel:(i+1)*N//channel,:])
        for j in range(evs.shape[0]):
            grids[evs[j,2].astype('int64'), evs[j,1].astype('int64'),i] += evs[j,3]
    return grids

def voxel_grid_equal_time(events,height=360,width=480,channel=3):
    interval = (events[-1,0]-events[0,0])//channel      
    N = events.shape[0]
    grids = np.zeros((height,width,channel))
    j = 0
    for i in range(channel):
        end_t = events[0,0]+(i+1)*interval
        while j<N and events[j,0] < end_t:
            grids[events[j,2].astype('int64'), events[j,1].astype('int64'),i] += events[j,3]
            j+=1
        
    return grids

class MixedDensityEventStacking:
    NO_VALUE = 0.
    STACK_LIST = ['stacked_polarity', 'index']

    def __init__(self, stack_size=3, height=360, width=480):
        self.stack_size = stack_size
        
        self.height = height
        self.width = width

    def pre_stack(self, event_sequence, last_timestamp):
        x = event_sequence[:,1].astype(np.int32)
        y = event_sequence[:,2].astype(np.int32)
        p = 2 * event_sequence[:,3].astype(np.int8) - 1
        t = event_sequence[:,0].astype(np.int64)

        assert len(x) == len(y) == len(p) == len(t)

        past_mask = t < last_timestamp
        p_x, p_y, p_p, p_t = x[past_mask], y[past_mask], p[past_mask], t[past_mask]
        p_t = p_t - p_t.min()
        past_stacked_event = self.make_stack(p_x, p_y, p_p, p_t)

        future_mask = t >= last_timestamp
        if np.sum(future_mask) == 0:
            stacked_event_list = [past_stacked_event]
        else:
            f_x = x[future_mask][::-1]
            f_y = y[future_mask][::-1]
            f_p = p[future_mask][::-1]
            f_t = t[future_mask][::-1]
            f_p = f_p * -1
            f_t = f_t - f_t.min()
            f_t = f_t.max() - f_t
            future_stacked_event = self.make_stack(f_x, f_y, f_p, f_t)

            stacked_event_list = [past_stacked_event, future_stacked_event]

        return stacked_event_list

    def post_stack(self, pre_stacked_event):
        stacked_event_list = []
        #print('pre_stacked_event',len(pre_stacked_event))
        for pf_stacked_event in pre_stacked_event:
            stacked_polarity = np.zeros([self.height, self.width, 1], dtype=np.float32)
            cur_stacked_event_list = []
            for stack_idx in range(self.stack_size - 1, -1, -1):
                stacked_polarity.put(pf_stacked_event['index'][stack_idx],
                                     pf_stacked_event['stacked_polarity'][stack_idx])
                cur_stacked_event_list.append(np.stack([stacked_polarity], axis=2))
                #print('cur_stacked_list',cur_stacked_event_list[0].shape)
            stacked_event_list.append(np.concatenate(cur_stacked_event_list[::-1], axis=2))
            #print('stacked_event_list',stacked_event_list[0].shape)
        if len(stacked_event_list) == 2:
            stacked_event_list[1] = stacked_event_list[1][:, :, ::-1, :]
        stacked_event = np.stack(stacked_event_list, axis=2)
        #print('stacked_event_list',len(stacked_event_list))
        #print('event_stack',stacked_event.shape)
        return stacked_event

    def make_stack(self, x, y, p, t):
        t = t - t.min()
        time_interval = t.max() - t.min() + 1
        t_s = (t / time_interval * 2) - 1.0
        stacked_event_list = {stack_value: [] for stack_value in self.STACK_LIST}
        cur_num_of_events = len(t)
        for _ in range(self.stack_size):
            stacked_event = self.stack_data(x, y, p, t_s)
            stacked_event_list['stacked_polarity'].append(stacked_event['stacked_polarity'])

            cur_num_of_events = cur_num_of_events // 2
            x = x[cur_num_of_events:]
            y = y[cur_num_of_events:]
            p = p[cur_num_of_events:]
            t_s = t_s[cur_num_of_events:]
            t = t[cur_num_of_events:]

        grid_x, grid_y = np.meshgrid(np.linspace(0, self.width - 1, self.width, dtype=np.int32),
                                     np.linspace(0, self.height - 1, self.height, dtype=np.int32))
        for stack_idx in range(self.stack_size - 1):
            prev_stack_polarity = stacked_event_list['stacked_polarity'][stack_idx]
            next_stack_polarity = stacked_event_list['stacked_polarity'][stack_idx + 1]

            assert np.all(next_stack_polarity[(prev_stack_polarity - next_stack_polarity) != 0] == 0)

            diff_stack_polarity = prev_stack_polarity - next_stack_polarity

            mask = diff_stack_polarity != 0
            stacked_event_list['index'].append((grid_y[mask] * self.width) + grid_x[mask])
            stacked_event_list['stacked_polarity'][stack_idx] = diff_stack_polarity[mask]

        last_stack_polarity = stacked_event_list['stacked_polarity'][self.stack_size - 1]
        mask = last_stack_polarity != 0
        stacked_event_list['index'].append((grid_y[mask] * self.width) + grid_x[mask])
        stacked_event_list['stacked_polarity'][self.stack_size - 1] = last_stack_polarity[mask]

        return stacked_event_list

    def stack_data(self, x, y, p, t_s):
        assert len(x) == len(y) == len(p) == len(t_s)

        stacked_polarity = np.zeros([self.height, self.width], dtype=np.int8)

        index = (y * self.width) + x

        stacked_polarity.put(index, p)

        stacked_event = {
            'stacked_polarity': stacked_polarity,
        }

        return stacked_event
    
    # @staticmethod
    # def collate_fn(batch):
    #     batch = torch.utils.data._utils.collate.default_collate(batch)

    #     return batch
    def main_process(self,events):
        # print('in main_process')
        event_data = self.pre_stack(events,last_timestamp=events[-1,0]+1)
        # print('pre',event_data)
        event_data = self.post_stack(event_data)
        # print('post',event_data.shape)
        return np.squeeze(event_data)