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

def measure_distance(p1,p2):
    """
    Measure the distance between two points

    Args:
        p1 (list): Point 1 in the format [x, y]
        p2 (list): Point 2 in the format [x, y] 

    Returns:
        float: Distance between the two points    
    """
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5 

def measure_xy_distance(p1,p2):
    """
    Measure the distance between two points in x and y axis

    Args:
        p1 (list): Point 1 in the format [x, y]
        p2 (list): Point 2 in the format [x, y]

    Returns:    
        float: Distance between the two points in x axis
        float: Distance between the two points
    """
    return p1[0] - p2[0], p1[1] - p2[1]

def get_foot_position(bbox):
    """
    Get the foot position from the bounding box

    Args:
        bbox (list): Bounding box in the format [x1, y1, x2, y2]

    Returns:
        list: Foot position in the format [x, y]
    """
    x1, y1, x2, y2 = bbox
    return int((x1 + x2)/2), int(y2)