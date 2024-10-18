def get_center_off_bbox(bbox):
    """
    Get the center of a bounding box

    Args:
        bbox (list): Bounding box in the format [x1, y1, x2, y2]

    Returns:
        list: Center of the bounding box in the format [x, y]
    """
    x1, y1, x2, y2 = bbox
    return int((x1 + x2)/2), int((y1 + y2)/2)

def get_bbox_width(bbox):
    """
    Get the width of a bounding box

    Args:
        bbox (list): Bounding box in the format [x1, y1, x2, y2]

    Returns:
        float: Width of the bounding box
    """
    return bbox[2] - bbox[0]