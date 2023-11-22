import torch

EPSILON = 1e-8

def polgyon_area(polygons):
    """
    Calculate the area of a poygon via Shoelace algorithm
 
    The input polygon is a tensor of its vertices with the last two dimension of shape (n, 2)
    where n is the number of vertices. The vertices is ordered clockwise.
    """

    # Roll the vertices to create cyclic pairs (x1, y1) -> (x2, y2), (x2, y2) -> (x3, y3), ...
    rolled_polygons = torch.roll(polygons, shifts=-1, dims=-2)

    # Calculate the area using the shoelace formula
    area = 0.5 * torch.abs(torch.sum(polygons[..., 0] * rolled_polygons[..., 1] -
                                     rolled_polygons[..., 0] * polygons[..., 1], dim=-1))

    return area

#########################################################################
# Find polygon intersection via Sutherland–Hodgman clipping algorithm
# This implementation is based on https://github.com/mhdadk/sutherland-hodgman
#########################################################################

# POINTS NEED TO BE PRESENTED CLOCKWISE OR ELSE THIS WONT WORK
def is_inside(p1, p2, q):
    a = p2 - p1
    b = q - p1
    return a[0] * b[1] - a[1] * b[0] >= 0

def intersect_lines(p1, p2, p3, p4):
    """
    given points p1 and p2 on line L1, compute the equation of L1 in the
    format of y = m1 * x + b1. Also, given points p3 and p4 on line L2,
    compute the equation of L2 in the format of y = m2 * x + b2.

    To compute the point of intersection of the two lines, equate
    the two line equations together

    m1 * x + b1 = m2 * x + b2

    and solve for x. Once x is obtained, substitute it into one of the
    equations to obtain the value of y.

    if one of the lines is vertical, then the x-coordinate of the point of
    intersection will be the x-coordinate of the vertical line. Note that
    there is no need to check if both lines are vertical (parallel), since
    this function is only called if we know that the lines intersect.
    """

    # if first line is vertical
    if p2[0] - p1[0] == 0:
        x = p1[0]
        # slope and intercept of second line
        m2 = (p4[1] - p3[1]) / (p4[0] - p3[0])
        b2 = p3[1] - m2 * p3[0]
        # y-coordinate of intersection
        y = m2 * x + b2

    # if second line is vertical
    elif p4[0] - p3[0] == 0:
        x = p3[0]
        # slope and intercept of first line
        m1 = (p2[1] - p1[1]) / (p2[0] - p1[0])
        b1 = p1[1] - m1 * p1[0]
        # y-coordinate of intersection
        y = m1 * x + b1

    # if neither line is vertical
    else:
        m1 = (p2[1] - p1[1]) / (p2[0] - p1[0])
        b1 = p1[1] - m1 * p1[0]
        # slope and intercept of second line
        m2 = (p4[1] - p3[1]) / (p4[0] - p3[0])
        b2 = p3[1] - m2 * p3[0]
        # x-coordinate of intersection
        x = (b2 - b1) / (m1 - m2)
        # y-coordinate of intersection
        y = m1 * x + b1

    # need to unsqueeze so torch.cat doesn't complain outside func
    intersection = torch.stack((x,y)).unsqueeze(0)

    return intersection

def intersect_polygons_clockwise(subject_polygon, clipping_polygon):
    clipped_polygon = torch.clone(subject_polygon)

    for i in range(len(clipping_polygon)):
        # stores the vertices of the next iteration of the clipping procedure
        cur_polygon = clipped_polygon

        # stores the vertices of the final clipped polygon. This will be
        # a K x 2 tensor, so need to initialize shape to match this
        clipped_polygon = torch.empty((0,2)).to(subject_polygon.device)

        # these two vertices define a line segment (edge) in the clipping
        # polygon. It is assumed that indices wrap around, such that if
        # i = 0, then i - 1 = M.
        c_start = clipping_polygon[i - 1]
        c_end = clipping_polygon[i]

        for j in range(len(cur_polygon)):

            # these two vertices define a line segment (edge) in the subject
            # polygon
            s_start = cur_polygon[j - 1]
            s_end = cur_polygon[j]

            if is_inside(c_start,c_end,s_end):
                if not is_inside(c_start,c_end,s_start):
                    intersection = intersect_lines(s_start,s_end,c_start,c_end)
                    clipped_polygon = torch.cat((clipped_polygon,intersection),dim=0)
                clipped_polygon = torch.cat((clipped_polygon,s_end.unsqueeze(0)),dim=0)
            elif is_inside(c_start,c_end,s_start):
                intersection = intersect_lines(s_start,s_end,c_start,c_end)
                clipped_polygon = torch.cat((clipped_polygon,intersection),dim=0)

    return clipped_polygon

def polygon_iou(polygons_a, polygons_b, use_giou=False):
    """
    Dimension of input polygons : (B, N, V, 2)
        where B is the batch size, N and V is the number of polygons and vertices
    """
    areas_a, areas_b = polgyon_area(polygons_a), polgyon_area(polygons_b)
    if use_giou:
        c_tl = torch.min(torch.amin(polygons_a, dim=-2), torch.amin(polygons_b, dim=-2))
        c_br = torch.max(torch.amax(polygons_a, dim=-2), torch.amax(polygons_b, dim=-2))
        area_c = torch.prod(c_br - c_tl, -1)

    B, N = polygons_a.shape[:2]
    ious = torch.zeros((B, N), dtype=torch.float32)
    for b in range(B):
        for n in range(N):
            polygon_i = intersect_polygons_clockwise(polygons_a[b, n], polygons_b[b, n])
            area_i = polgyon_area(polygon_i)
            area_u = areas_a[b, n] + areas_b[b, n] - area_i
            iou = area_i / (area_u + EPSILON)
            if use_giou:
                # compute giou
                iou = iou - (area_c[b, n] - area_u) / area_c[b, n].clamp(EPSILON)

            ious[b, n] = iou

    return ious


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    import numpy as np

    torch.autograd.set_detect_anomaly(True)

    polygons_p = torch.tensor([[[[20, 20], [40, 80], [90, 50], [60, 10]]]], dtype=torch.float32,
                             requires_grad=True)
    polygons_t = torch.tensor([[[[15, 30], [40, 10], [90, 30], [50, 80]]]], dtype=torch.float32)

    # polygon IOU
    iou = polygon_iou(polygons_p, polygons_t)

    # IOU loss
    iou = iou.mean(dim=-1)
    loss_iou = 1 - iou ** 2
    loss_iou = loss_iou.sum()

    # verify differentiability
    loss_iou.backward()
    print(f"Predicted polygon grad: {polygons_p.grad}")

    # Plot the polygons and their intersection
    polygon_p, polygon_t = polygons_p[0, 0], polygons_t[0, 0]
    polygon_i = intersect_polygons_clockwise(polygon_p, polygon_t)

    image_shape = (100, 100)
    # Create a blank image with the specified shape
    image = torch.zeros(image_shape, dtype=torch.uint8)

    # Draw the polygon on the visualization
    plt.figure(figsize=(8, 8))
    polygon_patch = Polygon(polygon_p.detach().numpy() , edgecolor='black', fill=None, linewidth=2)
    plt.gca().add_patch(polygon_patch)
    polygon_patch = Polygon(polygon_t.detach().numpy() , edgecolor='blue', fill=None, linewidth=2)
    plt.gca().add_patch(polygon_patch)
    polygon_patch = Polygon(polygon_i.detach().numpy() , edgecolor='red', fill=None, linewidth=2)
    plt.gca().add_patch(polygon_patch)

    overlay_mask = np.zeros_like(image.numpy(), dtype=np.uint8)
    plt.imshow(overlay_mask, cmap='tab20c')
    plt.title('Overlay of Polygons')
    plt.show()
    plt.close()
