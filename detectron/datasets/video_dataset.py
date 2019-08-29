import glob
import cv2
import pdb

class Dataset_from_videos:
    def __init__(self, avi_list):
        assert isinstance(avi_list, list), 'avi_list should be a list!'
        assert len(avi_list) > 0, 'Pass videofiles!'
        self.avi_list = avi_list
        self._cur_index = -1
        self._cur_reader = None
        self._cur_avi_file = None
        self._side = 'R'
        self._frame = None
        self._framerates = []
        self._end = False
        # self.frame = 0 # 1-based frame count, per video file

        # Set total number of frames
        tot_count = 0
        for avi_file in self.avi_list:
            tot_count += int(cv2.VideoCapture(avi_file).get(cv2.CAP_PROP_FRAME_COUNT))
            self._framerates.append(cv2.VideoCapture(avi_file).get(cv2.CAP_PROP_FPS))
            # tot_count += int(cv2.VideoCapture(avi_file).get(cv2.cv.CAP_PROP_FRAME_COUNT))
        self.tot_frames = tot_count * 2 # Due to left and right image

    def get_frame_shape(self):
        return self.get_frame(0, 'L').shape

    def next_video(self):
        if self._side == 'L':
            # Switch side
            self._side = 'R'
            # Reset cursor to start of video
            self.cur_reader.set(cv2.CAP_PROP_POS_AVI_RATIO, 0)
        else:
            # To next video
            if (self._cur_index + 1) == len(self.avi_list):
                self._end = True
            else:
                self._cur_index += 1
                self._cur_avi_file = self.avi_list[self._cur_index]
                self._cur_reader = cv2.VideoCapture(self._cur_avi_file)
                self._side = 'L'

    def get_cur_avi_file(self):
        if self._cur_avi_file is None:
            self.next_video()
            if self._end:
                return None
        return self._cur_avi_file

    def get_cur_reader(self):
        if self._cur_reader is None:
            self.next_video()
            if self._end:
                return None
        return self._cur_reader

    def get_next_frame(self, return_frame=True):
        if return_frame:
            ret, frame = self.cur_reader.read()
            if not ret:
                self.next_video()
                if self._end:
                    return None, None
                ret, frame = self.cur_reader.read()

            image = frame[:, :int(frame.shape[1] / 2), :] if self._side == 'L' else frame[:, int(frame.shape[1] / 2):, :]
            metadata = {
                        'frame': self._cur_reader.get(cv2.CAP_PROP_POS_FRAMES),
                        'video_file': self._cur_avi_file,
                        'side': self._side,
                        'frame_id': '_'.join([str(self._cur_index), str(int(self._cur_reader.get(cv2.CAP_PROP_POS_FRAMES)))]),
                        'frame_width': self._cur_reader.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH),
                        'frame_height': self._cur_reader.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
                       }
            return image, metadata
        else:
            ret = self.cur_reader.grab()
            if not ret:
                self.next_video()
                if self._end:
                    return None
                ret = self.cur_reader.grab()
            else:
                metadata = {
                            'frame': self._cur_reader.get(cv2.CAP_PROP_POS_FRAMES),
                            'video_file': self._cur_avi_file,
                            'side': self._side,
                            'frame_id': '_'.join([str(self._cur_index), str(int(self._cur_reader.get(cv2.CAP_PROP_POS_FRAMES)))]),
                            'frame_width': self._cur_reader.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH),
                            'frame_height': self._cur_reader.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
                           }
                return metadata

    def skip_to_frame(self, frame):
        for i in range(frame):
            self.get_next_frame(return_frame=False)

    def get_frame_for_frame_id(self, frameid):
        '''
        frameids are formatted as [file_index]_[frame #]
        '''
        cursor, framenumber = [int(x) for x in frameid.split('_')]
        reader = cv2.VideoCapture(self.avi_list[cursor])
        reader.set(cv2.CAP_PROP_POS_FRAMES, framenumber - 1)
        ret, frame = reader.read()
        if ret:
            return frame
        else:
            return None

    def get_frame(self, framenumber, side):
        '''
        Works only for the current videofile (first if not specified)
        '''
        self.cur_reader.set(cv2.CAP_PROP_POS_FRAMES, framenumber - 1)
        ret, frame = self.cur_reader.read()
        if frame is None:
            pdb.set_trace()
        image = frame[:, :int(frame.shape[1] / 2), :] if side == 'L' else frame[:, int(frame.shape[1] / 2):, :]
        # Convert colorspace
        rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        return rgb

    cur_reader = property(fget=get_cur_reader)
    cur_avi_file = property(fget=get_cur_avi_file)
