from sklearn.preprocessing import StandardScaler
import numpy as np

# file_names = ["harald_music", "harald_no_music", "jacob_cartwheel", "jacob_clean_jerk", "jacob_dance",
#                   "jacob_snatch"]

bike_names = ["rec7_valerio_bike0010_ref","rec7_valerio_bike0011_asym","rec7_valerio_bike0012_shoulders",
              "rec7_valerio_bike0013_asynch"]

#100 FPS
def motion_load_IPEM(name, skip_rows,skip_columns):
    with open('data/'+name+".tsv") as tsvfile:
        all_text = [line.split('\t') for line in tsvfile.readlines()]

        header=all_text[skip_rows]
        # marker_names = header[1:]
        marker_values= np.array(all_text[skip_rows + 1:]).astype(float)[:,skip_columns:]
        marker_pos=marker_values.reshape(marker_values.shape[0],-1,3)
        #normalisation can be done in different ways and strongly affects the interface
        #average all joint positions according to 8,9 14,15 (shoulder and hips)
        # center_pos=(marker_pos[:,8,:]+marker_pos[:,9,:]+marker_pos[:,14,:]+marker_pos[:,15,:])/4
    return marker_pos,marker_pos.shape[1]

def center_norm_data(motion_xyz_data):
    center_pos= (motion_xyz_data[:, 8, :] + motion_xyz_data[:, 9, :] + motion_xyz_data[:, 14, :] + motion_xyz_data[:, 15, :]) / 4

    print(motion_xyz_data[:, 8, :])

    motion_xyz_data= motion_xyz_data - np.repeat(center_pos, motion_xyz_data.shape[1], axis=0).reshape((motion_xyz_data.shape[0], -1, 3))
    motion_xyz_data= (motion_xyz_data-np.min(motion_xyz_data)) / (np.max(motion_xyz_data)-np.min(motion_xyz_data))
    return motion_xyz_data


class IPEM_plot():

    def __init__(self,ax_mot=None,motion_xyz=None):
        #for dots on the joints
        # self.graph, = ax_mot.plot(*motion_xyz[0].T, linestyle="", marker="o",markersize=2)
        # lines for bones
        self.ax_mot=ax_mot
        self.bones_index = []
        self.plots = []
        if motion_xyz is not None:
            motion_xyz = motion_xyz.reshape(motion_xyz.shape[0], -1, 3)
            init_row = motion_xyz[0]
        parent_child_joint = dict([(16, [19, 14, 15]), (15, [12]), (14, [13]), (12, [10]), (13, [11]), (10, [0]), (11, [1]),
                                   (19, [17, 18, 22, 23]), (17, [9]), (18, [8]), (9, [6]), (8, [7]), (7, [4])
                                      , (6, [5]), (4, [3]), (22, [23])])
        for connection in list(parent_child_joint.items()):
            parent = connection[0]
            for child in connection[1]:
                self.bones_index.append((parent,child))
                if (ax_mot is not None):
                    p_pos = init_row[parent]
                    c_pos = init_row[child]
                    line = np.stack((p_pos, c_pos), axis=1)
                    # ax.plot returns list, the first element is the line
                    self.plots.append(ax_mot.plot(*line, lw=2, c="b")[0])
        # if a global view is set
        # min_coord = np.array([np.min(motion_xyz.reshape(-1, 3)[:, a]) for a in range(3)])
        # max_coord = np.array([np.max(motion_xyz.reshape(-1, 3)[:, a]) for a in range(3)])
        # ax_mot.set_xlim3d([min_coord[0], max_coord[0]])
        # ax_mot.set_zlim3d([min_coord[2], max_coord[2]])
        # ax_mot.set_ylim3d([min_coord[1], max_coord[1]])
        # ax_mot.set_aspect('equal')
    def update(self,motion_xyz_frame,radius=2):
        # motion
        # self.graph.set_data(*motion_xyz_frame.T[0:2])
        # self.graph.set_3d_properties(motion_xyz_frame.T[2])
        bones=[]
        motion_xyz_frame = motion_xyz_frame.reshape(-1, 3)
        for bone_index in self.bones_index:
            bone = [motion_xyz_frame[bone_index[0]],motion_xyz_frame[bone_index[1]]]
            bones.append(bone)
        if self.ax_mot  is not None:
            for line,bone in zip(self.plots,bones):
                line.set_xdata([bone[0][0], bone[1][0]])
                line.set_ydata([bone[0][1], bone[1][1]])
                line.set_3d_properties([bone[0][2], bone[1][ 2]])
        r = radius;
        if self.ax_mot is not None:
            xroot, yroot, zroot =np.average(motion_xyz_frame,axis=0)

            self.ax_mot.set_xlim3d([-r + xroot, r + xroot])
            self.ax_mot.set_zlim3d([-r + zroot, r + zroot])
            self.ax_mot.set_ylim3d([-r + yroot, r + yroot])

            # self.ax.set_aspect('equal')

            self.ax_mot.set_aspect('equal')
        return bones



