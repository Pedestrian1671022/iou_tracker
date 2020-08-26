def track_iou(detections, tracks_active, sigma_l, sigma_h, sigma_iou, t_life, t_loss, id):
    """
    Simple IOU based tracker.
    See "High-Speed Tracking-by-Detection Without Using Image Information by E. Bochinski, V. Eiselein, T. Sikora" for
    more information.
    Args:
         detections (list): list of detections per frame
         tracks_active (list): list of tracks_active
         sigma_l (float): low detection threshold.
         sigma_h (float): high detection threshold.
         sigma_iou (float): IOU threshold.
         t_life (float): minimum track length in frames.
    Returns:
        list: list of tracks_finished and tracks_active.
    """
    tracks_finished = []

    # apply low threshold to detections
    dets = [det for det in detections if det['score'] >= sigma_l]

    updated_tracks = []
    for track in tracks_active:
        if len(dets) > 0:
            # get det with highest iou
            best_match = max(dets, key=lambda x: iou(track['bboxes'], x['bbox']))
            if iou(track['bboxes'], best_match['bbox']) >= sigma_iou:
                track['bboxes'] = best_match['bbox']
                track['max_score'] = max(track['max_score'], best_match['score'])
                track['life_time'] += 1
                track['loss_time'] = 0
                updated_tracks.append(track)

                # remove from best matching detection from detections
                del dets[dets.index(best_match)]
                if track['max_score'] >= sigma_h and track['life_time'] >= t_life:
                    if track['id'] == 0:
                        track['id'] = id
                        id += 1
                    tracks_finished.append(track)

        # if track was not updated
        if len(updated_tracks) == 0 or track is not updated_tracks[-1]:
            track['loss_time'] += 1
            # finish track when the conditions are met
            if track['loss_time'] >= t_loss:
                tracks_active.remove(track)

    # create new tracks
    new_tracks = [{'bboxes': det['bbox'], 'max_score': det['score'], 'life_time':1, 'loss_time': 0, 'id': 0} for det in dets]
    tracks_active = tracks_active + new_tracks

    return tracks_finished, tracks_active, id


def iou(bbox1, bbox2):
    """
    Calculates the intersection-over-union of two bounding boxes.
    Args:
        bbox1 (numpy.array, list of floats): bounding box in format x1,y1,x2,y2.
        bbox2 (numpy.array, list of floats): bounding box in format x1,y1,x2,y2.
    Returns:
        int: intersection-over-onion of bbox1, bbox2
    """

    bbox1 = [float(x) for x in bbox1]
    bbox2 = [float(x) for x in bbox2]

    (x0_1, y0_1, x1_1, y1_1) = bbox1
    (x0_2, y0_2, x1_2, y1_2) = bbox2

    # get the overlap rectangle
    overlap_x0 = max(x0_1, x0_2)
    overlap_y0 = max(y0_1, y0_2)
    overlap_x1 = min(x1_1, x1_2)
    overlap_y1 = min(y1_1, y1_2)

    # check if there is an overlap
    if overlap_x1 - overlap_x0 <= 0 or overlap_y1 - overlap_y0 <= 0:
        return 0

    # if yes, calculate the ratio of the overlap to each ROI size and the unified size
    size_1 = (x1_1 - x0_1) * (y1_1 - y0_1)
    size_2 = (x1_2 - x0_2) * (y1_2 - y0_2)
    size_intersection = (overlap_x1 - overlap_x0) * (overlap_y1 - overlap_y0)
    size_union = size_1 + size_2 - size_intersection

    return size_intersection / size_union