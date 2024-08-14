import cv2
import numpy as np

def _get_color(idx):
    """
    Generate a color based on an index value.

    Args:
        idx (int): Index to generate a unique color.

    Returns:
        tuple: A tuple representing the BGR color.
    """
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color

def plot(
    frame:np.ndarray, tlwhs:list, obj_ids:list, names:list, bboxes:np.ndarray
):
    text_scale = 2
    text_thickness = 2
    line_thickness = 3
    if tlwhs != [] and tlwhs is not None: #is_tracker_available
        cv2.putText(
            frame,
            "num: %d" % (len(tlwhs)),
            (0, int(15 * text_scale)),
            cv2.FONT_HERSHEY_PLAIN,
            2,
            (0, 0, 255),
            thickness=2,
        )

        for i, tlwh in enumerate(tlwhs):
            x1, y1, w, h = tlwh
            intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
            obj_id = int(obj_ids[i])
            id_text = "{}".format(int(obj_id))
            if (obj_id) in names:
                id_text = id_text + ": " + names[obj_id]
            color = _get_color(abs(obj_id))
            cv2.rectangle(
                frame, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness
            )
            cv2.putText(
                frame,
                id_text,
                (intbox[0], intbox[1]),
                cv2.FONT_HERSHEY_PLAIN,
                text_scale,
                (0, 0, 255),
                thickness=text_thickness,
            )
        return frame

    h, w, c = frame.shape

    # Draw bounding boxes and landmarks on the frame
    for i in range(len(bboxes)):
        cv2.putText(
            frame,
            "num: %d" % (len(bboxes)),
            (0, int(15 * text_scale)),
            cv2.FONT_HERSHEY_PLAIN,
            2,
            (0, 0, 255),
            thickness=2,
        )
        if names == {}:
            x1, y1, x2, y2, score = bboxes[i]
            intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
            color = _get_color(i)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color=color, thickness=line_thickness)
            continue
        
        try:
            x1, y1, x2, y2, score = bboxes[i]
            intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
            color = _get_color(i)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color=color, thickness=line_thickness)

            if i < len(names):
                cv2.putText(
                    frame,
                    names[i],
                    (intbox[0], intbox[1]),
                    cv2.FONT_HERSHEY_PLAIN,
                    text_scale,
                    (0, 0, 255),
                    thickness=text_thickness,
                )
        except Exception as e:
            print(f"Error processing bounding boxes: {e}")


    return frame
