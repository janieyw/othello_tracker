import cv2
import numpy as np


class GridDetector:

    def __init__(self):
        self.img = None

    def set_image(self, img_path):
        self.img = cv2.imread(img_path)

    def detect_lines(self):
        # convert to grayscale and blur the image
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # detect edges using Canny
        edges = cv2.Canny(blur, 50, 150, apertureSize=3)

        # detect lines using HoughLinesP
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)

        return lines

    # Referenced https://stackoverflow.com/questions/45531074/how-to-merge-lines-after-houghlinesp
    def merge_lines_pipeline_2(self, lines, img_shape):
        'Connects detected lines of a grid'

        # Define the size of the image for distance calculation
        height, width = img_shape[:2]
        size = height * width

        # Define a threshold based on image size to merge the lines
        threshold = size * 0.01

        # Parameters to play with
        min_distance_to_merge = 30
        min_angle_to_merge = 30

        # Clusterize (group) lines
        groups = []  # all lines groups are here

        # first line will create new group every time
        groups.append([lines[0]])

        # if line is different from existing groups, create a new group
        for line_new in lines[1:]:
            if self.checker(line_new, groups, min_distance_to_merge, min_angle_to_merge):
                groups.append([line_new])

        # Merge lines within the same group
        for group in groups:
            merged_line = self.merge_lines_segments1(group)
            group.clear()
            group.append(merged_line)

        # Merge lines between different groups
        final_groups = []
        for i, group in enumerate(groups):
            if i == 0:
                final_groups.append(group)
                continue
            group_i_first = group[0][:2]
            group_i_last = group[0][2:]
            added_to_existing_group = False
            for final_group in final_groups:
                final_group_first = final_group[0][:2]
                final_group_last = final_group[0][2:]
                dist_first_to_last = np.linalg.norm(np.array(group_i_first) - np.array(final_group_last))
                dist_last_to_first = np.linalg.norm(np.array(group_i_last) - np.array(final_group_first))
                if dist_first_to_last < threshold:
                    final_group.append(group[0])
                    added_to_existing_group = True
                    break
                elif dist_last_to_first < threshold:
                    final_group.insert(0, group[0])
                    added_to_existing_group = True
                    break
            if not added_to_existing_group:
                final_groups.append(group)

        # Draw final groups
        for group in final_groups:
            color = np.random.randint(0, 255, size=3).tolist()
            for line in group:
                x1, y1, x2, y2 = line
                cv2.line(self.img, (x1, y1), (x2, y2), color, 2)

        return self.img

    # Referenced https://stackoverflow.com/questions/45531074/how-to-merge-lines-after-houghlinesp
    def merge_lines_segments1(self, lines):
        """Sort lines cluster and return first and
        return first and last coordinates of the sorted group of line segments"""
        # last coordinates
        orientation = self.get_orientation(lines[0])

        # special case
        if (len(lines) == 1):
            return [lines[0][:2], lines[0][2:]]

        # [[1,2,3,4],[]] to [[1,2],[3,4],[],[]]
        points = []
        for line in lines:
            points.append(line[:2])
            points.append(line[2:])

        # if vertical
        if 45 < orientation < 135:
            # sort by y
            points = sorted(points, key=lambda point: point[1])
        else:
            # sort by x
            points = sorted(points, key=lambda point: point[0])

        # return first and last point in sorted group
        # [[x,y],[x,y]]
        return [points[0], points[-1]]

    def get_orientation(self, line):
        """Calculate orientation of a line"""
        x1, y1, x2, y2 = line
        dx = x2 - x1
        dy = y2 - y1
        return np.degrees(np.arctan2(dy, dx))

    def checker(self, line_new, groups, min_distance_to_merge, min_angle_to_merge):
        """Check if line is different from existing groups"""
        for group in groups:
            line_existing = group[-1]
            # Distance between the endpoints of two lines
            dist1 = np.linalg.norm(np.array(line_new[:2]) - np.array(line_existing[:2]))
            dist2 = np.linalg.norm(np.array(line_new[2:]) - np.array(line_existing[2:]))
            # Distance between the first endpoint of line1 and the last endpoint of line2
            dist3 = np.linalg.norm(np.array(line_new[:2]) - np.array(line_existing[2:]))
            # Distance between the last endpoint of line1 and the first endpoint of line2
            dist4 = np.linalg.norm(np.array(line_new[2:]) - np.array(line_existing[:2]))
            angle1 = abs(self.get_orientation(line_new) - self.get_orientation(line_existing))
            angle2 = abs(self.get_orientation(line_new) - self.get_orientation(
                [line_existing[2], line_existing[3], line_existing[0], line_existing[1]]))
            if min(dist1, dist2, dist3, dist4) < min_distance_to_merge and min(angle1, angle2) < min_angle_to_merge:
                group.append(line_new)
                return False
        return True

