import numpy as np
import cv2


def line_intersection(line1, line2):
x1, y1, x2, y2 = (
line1["x1"],
line1["y1"],
line1["x2"],
line1["y2"]
)

x3, y3, x4, y4 = (
line2["x1"],
line2["y1"],
line2["x2"],
line2["y2"]
)

denominator = (
(x1 - x2) * (y3 - y4)
- (y1 - y2) * (x3 - x4)
)

if abs(denominator) < 0.001:
return None

px = (
((x1 * y2 - y1 * x2) * (x3 - x4)
- (x1 - x2) * (x3 * y4 - y3 * x4))
/ denominator
)

py = (
((x1 * y2 - y1 * x2) * (y3 - y4)
- (y1 - y2) * (x3 * y4 - y3 * x4))
/ denominator
)

return int(px), int(py)


def get_court_corners(frame):

if frame is None:
return None

try:

height, width = frame.shape[:2]

gray = cv2.cvtColor(
frame,
cv2.COLOR_BGR2GRAY
)

blurred = cv2.GaussianBlur(
gray,
(5, 5),
0
)

edges = cv2.Canny(
blurred,
50,
150
)

lines_p = cv2.HoughLinesP(
edges,
rho=1,
theta=np.pi / 180,
threshold=100,
minLineLength=100,
maxLineGap=30
)

if lines_p is None:
return None

lines = []

for line in lines_p:

x1, y1, x2, y2 = line[0]

lines.append(
{
"x1": int(x1),
"y1": int(y1),
"x2": int(x2),
"y2": int(y2)
}
)

intersections = []

for i in range(len(lines)):

for j in range(i + 1, len(lines)):

point = line_intersection(
lines[i],
lines[j]
)

if point is None:
continue

x, y = point

if (
0 <= x < width
and 0 <= y < height
):
intersections.append(point)

if len(intersections) < 4:
return None

points = np.array(
intersections,
dtype=np.float32
)

# Convex hull finds outer boundary
hull = cv2.convexHull(points)

if len(hull) < 4:
return None

# Approximate polygon
epsilon = 0.02 * cv2.arcLength(
hull,
True
)

approx = cv2.approxPolyDP(
hull,
epsilon,
True
)

points = approx.reshape(-1, 2)

if len(points) < 4:

# Fallback bounding rectangle
x, y, w, h = cv2.boundingRect(hull)

return {
"top_left": (x, y),
"top_right": (x + w, y),
"bottom_right": (x + w, y + h),
"bottom_left": (x, y + h)
}

# If more than 4 points, use bounding rectangle
if len(points) != 4:

x, y, w, h = cv2.boundingRect(points)

points = np.array([
[x, y],
[x + w, y],
[x + w, y + h],
[x, y + h]
])

# Sort corners
points = points.astype(np.int32)

sorted_by_y = points[np.argsort(points[:, 1])]

top = sorted_by_y[:2]
bottom = sorted_by_y[2:]

top = top[np.argsort(top[:, 0])]
bottom = bottom[np.argsort(bottom[:, 0])]

top_left = tuple(top[0])
top_right = tuple(top[1])

bottom_left = tuple(bottom[0])
bottom_right = tuple(bottom[1])

return {
"top_left": top_left,
"top_right": top_right,
"bottom_right": bottom_right,
"bottom_left": bottom_left
}

except Exception as e:

print(
f"Court corner detection error: {e}"
)

return None
