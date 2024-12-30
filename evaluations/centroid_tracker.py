import numpy as np

class CentroidTracker():
    def __init__(self):
        self.tracked_obj = {
            'ids' : [],
            'centroids' : [],
            'missing_frame_count' : []
        }
        # self.img_size = img_size

        # Thresholds
        self.max_dist_thresh = 35
        self.max_missing_frame_count = 3
        
        self.total_id_count = 0

    def distance(self, x1, y1, x2, y2):
        return np.sqrt((x2-x1)**2 + (y2-y1)**2)

    def update(self, bbox_list):
        matched_ids = []
        tracked_bboxes = []

        tracked_obj = self.tracked_obj.copy()

        for bbox in bbox_list:
            center_x = min(bbox[0], bbox[2]) + ((max(bbox[0], bbox[2]) - min(bbox[0], bbox[2])) / 2)
            center_y = min(bbox[1], bbox[3]) + ((max(bbox[1], bbox[3]) - min(bbox[1], bbox[3])) / 2)

            closest = None
            closest_id = None
            for i in range(len(tracked_obj['ids'])):
                id = tracked_obj['ids'][i]
                centroid = tracked_obj['centroids'][i]

                if id in matched_ids:
                    continue

                dist = self.distance(center_x, center_y, centroid[0], centroid[1])

                print(dist)

                if dist < self.max_dist_thresh:
                    if closest == None or dist < closest:
                        closest = dist
                        closest_id = id

            if closest == None:
                self.total_id_count += 1
                new_id = self.total_id_count

                matched_ids.append(new_id)
                tracked_bboxes.append(bbox)

                self.tracked_obj['ids'].append(new_id)
                self.tracked_obj['centroids'].append([center_x, center_y])
                self.tracked_obj['missing_frame_count'].append(0)
            else:
                print(closest)
                matched_ids.append(closest_id)
                tracked_bboxes.append(bbox)

                idx = self.tracked_obj['ids'].index(closest_id)

                self.tracked_obj['centroids'][idx] = [center_x, center_y]
                self.tracked_obj['missing_frame_count'][idx] = 0

        for id in [i for i in self.tracked_obj['ids'] if not i in matched_ids]:
            idx = self.tracked_obj['ids'].index(id)
            self.tracked_obj['missing_frame_count'][idx] += 1

        for idx in [idx for idx, i in enumerate(self.tracked_obj['missing_frame_count']) if i > self.max_missing_frame_count]:
            del self.tracked_obj['ids'][idx]
            del self.tracked_obj['centroids'][idx]
            del self.tracked_obj['missing_frame_count'][idx]

        return {'tracked_bbox': tracked_bboxes, 'tracked_id': matched_ids}