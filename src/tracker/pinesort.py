from tracker.gmc import GMC
from tracker.track import KalmanBoxTracker
from tracker.utils import k_previous_obs, plot_and_save_image_with_detections, \
                    merge_indexes, merge_indexes_matches, is_out_margin

import tracker.matching as matching
import numpy as np
import logging

ASSO_FUNCS = { 
    "iou": matching.iou_batch,
    "giou": matching.giou_batch,
    "ciou": matching.ciou_batch,
    "diou": matching.diou_batch,
    "eiou": matching.eiou_batch
}

class PineSORT(object):
    def __init__(self, 
                 det_thresh=0.40, 
                 min_det_thresh=0.30,
                 max_age=5, 
                 min_hits=1, 
                 first_iou_threshold=0.30,
                 second_iou_threshold=0.30,
                 third_iou_threshold=0.10,
                 overlap_iou_threshold=0.10,
                 delta_t=3, 
                 inertia=0.2, 
                 asso_func="eiou",
                 camera_compensation="orb",
                 use_byte=True, img_width=1920, img_height=1088):
        
        self.max_age = max_age
        self.min_hits = min_hits
        self.first_iou_threshold = first_iou_threshold
        self.second_iou_threshold = second_iou_threshold
        self.third_iou_threshold = third_iou_threshold
        self.overlap_iou_threshold = overlap_iou_threshold
        self.trackers = []  # List to hold active trackers
        self.frame_count = 0  # Frame counter for tracking
        self.det_thresh = det_thresh
        self.min_det_thresh = min_det_thresh
        self.delta_t = delta_t
        self.inertia = inertia
        self.use_byte = use_byte  # Flag for using BYTE association
        self.asso_func = ASSO_FUNCS[asso_func]  # Association function lookup
        self.img_width = img_width
        self.img_height = img_height
        self.asso_name = asso_func
        self.camera_compensation = camera_compensation

        KalmanBoxTracker.count = 0  # Initialize KalmanBoxTracker count

        # Initialize the GMC (Global Motion Compensation) with a specific method
        self.gmc = GMC(method=self.camera_compensation)
        
    def update(self, detections, raw_img, path_save_debug=""):
        """
        Update the tracker state with the current frame's detections and raw image.
        
        Args:
            detections (np.ndarray): Array of detections for the current frame. Each detection is expected 
                                    to have bounding box coordinates and confidence scores.
            raw_img (np.ndarray): Raw image from the current frame, used for global motion compensation.

        Returns:
            np.ndarray: An array of processed detections with shape `(N, 5)`, where N is the number 
                        of final detections, and each row contains [x1, y1, x2, y2, score].
        """
        # Increment the frame counter
        self.frame_count += 1

        # Handle the case where no detections are provided
        if detections is None:
            return np.empty((0, 5))

        # Separate scores and bounding boxes from the detections
        if detections.shape[1] == 5:
            dets_scores = detections[:, 4]  # Extract detection scores
            dets_bboxes = detections[:, :4]  # Extract bounding box coordinates
        else:
            dets_scores = detections[:, 4] * detections[:, 5]  # Combine multiple scores
            dets_bboxes = detections[:, :4]  # Extract bounding box coordinates

        dets_bboxes = np.concatenate((dets_bboxes, np.expand_dims(dets_scores, axis=-1)), axis=1)  # Combine bboxes and scores

        # Identify detections with scores in specific ranges
        inds_low_score = dets_scores > self.min_det_thresh  # Scores above the minimum threshold
        inds_high_score = dets_scores < self.det_thresh  # Scores below the detection threshold
        inds_second = np.logical_and(inds_low_score, inds_high_score)  # Intermediate scores

        # Filter detections for second matching stage
        dets_second = dets_bboxes[inds_second]
        remain_inds = dets_scores > self.det_thresh  # Detections exceeding the threshold
        dets_bboxes = dets_bboxes[remain_inds]

        # Apply global motion compensation to adjust detections
        warp = self.gmc.apply(raw_img, dets_bboxes)
        self.trackers = KalmanBoxTracker.multi_gmc_xy(self.trackers, warp)

        # Prepare for tracker updates
        tracks = np.zeros((len(self.trackers), 5))  # Initialize array for tracker states
        to_delete_tracks = []  # List to store trackers marked for deletion
        return_tracks = []  # List to store final detections for output

        # Predict and update tracker state
        for t, track in enumerate(tracks):
            position = self.trackers[t].predict()[0]  # Predict next position using Kalman filter
            track[:] = [position[0], position[1], position[2], position[3], self.trackers[t].score]  # Update tracker
            if np.any(np.isnan(position)):  # Check for invalid positions
                to_delete_tracks.append(t)  # Mark tracker for removal

        # Save the figure
        plot_and_save_image_with_detections(raw_img, tracks, dets_bboxes, path_save_debug)

        # Remove invalid tracker states
        tracks = np.ma.compress_rows(np.ma.masked_invalid(tracks))

        # Delete trackers marked for removal
        for delete_track in reversed(to_delete_tracks):
            self.trackers.pop(delete_track)

        # Extract velocities and other features from trackers
        velocities_tracks = np.array([trk.velocity if trk.velocity is not None else np.array((0, 0)) for trk in self.trackers])
        last_boxes_tracks = np.array([trk.last_observation for trk in self.trackers])
        k_observations_tracks = np.array([k_previous_obs(trk.observations, trk.age, self.delta_t) for trk in self.trackers])

        """
        First round of association: Match detections with trackers
        """        
        # Using OCM from OCSORT did not have effect on the tracker accuracy
        matched, unmatched_dets, unmatched_tracks = matching.associate(
                dets_bboxes, tracks, self.first_iou_threshold, velocities_tracks, k_observations_tracks, self.inertia, association_fun=self.asso_name)

        for m_index in matched:
            self.trackers[m_index[1]].update(dets_bboxes[m_index[0], :], confidence=dets_bboxes[m_index[0], -1])
        
        """
        Second round of association
        """
        to_remove_dets_second_indices = []
        # If using BYTE association and there are low-confidence detections
        if self.use_byte and len(dets_second) > 0 and unmatched_tracks.shape[0] > 0:

            u_trks = tracks[unmatched_tracks]  # Get unmatched trackers
            iou_left = self.asso_func(dets_second, u_trks)  # Calculate IoU for low-confidence detections
            iou_left = np.array(iou_left)

            if iou_left.max() > self.second_iou_threshold:
                # Use linear assignment to rematch detections and trackers
                matched_indices = matching.linear_assignment(-iou_left)#iou_left)
                to_remove_trk_indices = []  # Track indices to remove
                for m in matched_indices:
                    det_ind, trk_ind = m[0], unmatched_tracks[m[1]]
                    if iou_left[m[0], m[1]] < self.second_iou_threshold: 
                        continue  # Skip if IoU is below threshold
                    else:
                        self.trackers[trk_ind].update(dets_second[det_ind, :], confidence=dets_second[det_ind, -1])  # Update tracker

                        to_remove_dets_second_indices.append(det_ind)
                        to_remove_trk_indices.append(trk_ind)  # Mark tracker for removal
                
                unmatched_tracks = np.setdiff1d(unmatched_tracks, np.array(to_remove_trk_indices))  # Remove matched trackers
                unmatched_dets = np.setdiff1d(unmatched_dets, np.array(to_remove_dets_second_indices))  # Remove matched trackers

                dets_second = np.delete(dets_second, to_remove_dets_second_indices, axis=0)

        """
        Combine dets_second and dets_bboxes
        """
        # This combine the not matched detections from the first and second association step given the threshold.
        if len(unmatched_dets) == 0:
            dets_bboxes = np.empty((0, 5))
        else:
            dets_bboxes = dets_bboxes[unmatched_dets]
            dets_bboxes = np.array(dets_bboxes)
        
        if len(dets_second) == 0:
            dets_second = np.empty((0, 5))
        else:
            dets_second = np.array(dets_second)

        dets_bboxes = np.concatenate((dets_bboxes, dets_second), axis=0)
        unmatched_dets = np.array(list(range(len(dets_bboxes))))

        """
        Third association to macth partial objects exiting the scenes
        """
        if unmatched_dets.shape[0] > 0 and unmatched_tracks.shape[0] > 0:
            left_dets = dets_bboxes[unmatched_dets]  # Remaining detections
            left_trks = tracks[unmatched_tracks]  # Remaining trackers

            iou_left = matching.diou_batch(left_dets, left_trks) # Normalized dIoU

            iou_left = np.array(iou_left)

            if iou_left.max() > self.third_iou_threshold: #self.iou_threshold:
                # Use linear assignment to rematch unmatched detections and trackers
                rematched_indices = matching.linear_assignment(-iou_left)
                to_remove_det_indices = []  # Track indices to remove
                to_remove_trk_indices = []  # Track indices to remove
                for m in rematched_indices:
                    det_ind, trk_ind = unmatched_dets[m[0]], unmatched_tracks[m[1]]
                    if iou_left[m[0], m[1]] < self.third_iou_threshold: #self.iou_threshold:
                        continue  # Skip if IoU is below threshold
                    
                    self.trackers[trk_ind].update(dets_bboxes[det_ind, :], confidence=dets_bboxes[m[0], -1])  # Update tracker
                    to_remove_det_indices.append(det_ind)  # Mark detection for removal
                    to_remove_trk_indices.append(trk_ind)  # Mark tracker for removal

                unmatched_dets = np.setdiff1d(unmatched_dets, np.array(to_remove_det_indices))  # Remove matched detections
                unmatched_tracks = np.setdiff1d(unmatched_tracks, np.array(to_remove_trk_indices))  # Remove matched trackers
                
        """
        Update unmatched trackers to indicate no detections
        """
        for m in unmatched_tracks:
            self.trackers[m].update(None)

        """
        Create and initialize new trackers for unmatched detections
        """
        new_trackers = []

        for i in unmatched_dets:
            """
            Overlapping management 
            Check if the new unmatched detection is overlapping with any Kalman Filter box, if match this means is a redundant detection.
            Given in the pineapple agriculture settings, on the field objects are spread in the space depending on the dron level to capture the objects.
            """ 
            value = matching.compute_overlap_tracks(self.trackers, dets_bboxes[i, :], overlap_threshold=self.overlap_iou_threshold)

            if value:
                continue 
            else:
                trk = KalmanBoxTracker(dets_bboxes[i, :], score=dets_bboxes[i,-1], delta_t=self.delta_t, kf_name="Fading")  # Create a new tracker with the Kalman Filter object
                new_trackers.append(trk)
                
        for new_track in new_trackers:
            self.trackers.append(new_track)  # Add to active trackers

        i = len(self.trackers)  # Get the count of active trackers

        # Prepare the output detections for valid trackers
        for trk in reversed(self.trackers):
            if trk.last_observation.sum() < 0:
                d = trk.get_state()[0][:4]  # Use predicted state if last observation is invalid
            else:
                # Optionally use the recent observation or Kalman filter prediction
                d = trk.last_observation[:4]  # Get last observation (bounding box)
            
            i -= 1  # Decrement index for removal of dead tracklet

            # Check if tracker is valid for output
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                return_tracks.append(np.concatenate((d, [trk.id+1], [trk.score+1])).reshape(1, -1))  # Append detection with ID
            
            #i -= 1  # Decrement index for removal of dead tracklet
            # Remove dead tracklet if it exceeds max_age
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        # Return updated detections or empty if none
        if len(return_tracks) > 0:
            return np.concatenate(return_tracks)  # Concatenate and return valid detections
        return np.empty((0, 5))  # Return empty if no valid detections