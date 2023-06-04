from Star import *
import numpy as np
from scipy.spatial import KDTree
import math
from PIL import Image, ImageDraw
from math import sqrt
from AffineTransform import getAffineTransform, dist_method


class Algorithm:
    def __init__(self):
        self.min_area = 1
        self.max_area = 1000

    def find_nearest_neighbors(self, query_point, points, k=2):
        """Find nearest neighbors of point and a set of points"""
        kdtree = KDTree(points)
        distances, indices = kdtree.query(query_point, k=k)
        return indices.tolist()

    def get_angle(self, points):
        """Get angle of the lines of p1"""
        p1 = dist_method(points[0], points[1])
        p2 = dist_method(points[1], points[2])
        p3 = dist_method(points[2], points[0])
        angle = math.acos((p1**2 + p2**2 - p3**2) / (2 * p1 * p2))
        return math.degrees(angle)

    def detect_nanopi(self, img):
        original_image = Image.open(img)
        image = original_image.resize((600, 600))
        width, height = image.size
        image = image.convert("L")
        threshold = 150
        binary_image = []
        for y in range(height):
            row = []
            for x in range(width):
                pixel_value = image.getpixel((x, y))
                row.append(1 if pixel_value > threshold else 0)
            binary_image.append(row)

        coordinates = []
        for y in range(height):
            for x in range(width):
                if binary_image[y][x] == 1:
                    if (
                        (y > 0 and binary_image[y - 1][x] == 0)
                        or (y < height - 1 and binary_image[y + 1][x] == 0)
                        or (x > 0 and binary_image[y][x - 1] == 0)
                        or (x < width - 1 and binary_image[y][x + 1] == 0)
                    ):
                        if all(
                            sqrt((x - prev_x) ** 2 + (y - prev_y) ** 2) >= 15
                            for prev_x, prev_y in coordinates
                        ):
                            coordinates.append((x, y))

        stars = []
        for coord in coordinates:
            stars.append(Star(x=coord[0], y=coord[1], brightness=0, radius=0))

        draw = ImageDraw.Draw(image)
        circle_radius = 10  # Adjust the radius as needed

        for coord in coordinates:
            x, y = coord
            draw.ellipse(
                [
                    (x - circle_radius, y - circle_radius),
                    (x + circle_radius, y + circle_radius),
                ],
                outline="white",
            )
        image.save(img + "output.png")
        return stars

    def draw_results(self, img, stars, image_name):
        """Method that draws the result of the match on the original imamge"""
        print("Drawing results")
        original_image = Image.open(img)
        image = original_image.resize((600, 600))
        image = image.convert("L")
        draw = ImageDraw.Draw(image)
        circle_radius = 10
        for coord in stars:
            draw.ellipse(
                [
                    (coord[0] - circle_radius, coord[1] - circle_radius),
                    (coord[0] + circle_radius, coord[1] + circle_radius),
                ],
                outline="white",
            )
        image.save(image_name)

    def stars_list_to_array(self, stars):
        list = []
        for star in stars:
            list.append((star.x, star.y))
        return list

    def random_sample(self, stars, num_samples):
        index = np.random.choice(len(stars), num_samples, replace=False)
        return [stars[i] for i in index]

    def make_transform(self, src_stars, dst_stars):
        src_pts = np.float32(src_stars)
        dst_pts = np.float32(dst_stars)
        matrix = getAffineTransform(src_pts, dst_pts)
        return matrix

    def check_inliers(
        self, src_pts_original, dst_pts_original, matrix, threshold, stars1, stars2
    ):
        """Check the inliers of src to dst"""
        inliers = []
        src_inliers = []
        src_pts = list(src_pts_original)
        dst_pts = list(dst_pts_original)
        len1 = len(src_pts)
        len2 = len(dst_pts)
        for i in range(len1):
            src_pt = np.array([src_pts[i][0], src_pts[i][1], 1], dtype=np.float32)
            len2 = len(dst_pts)
            for j in range(len2):
                dst_pt = np.array([dst_pts[j][0], dst_pts[j][1], 1], dtype=np.float32)
                transform_pt = np.dot(matrix, src_pt)
                dist = np.sqrt(
                    (dst_pt[0] - transform_pt[0]) ** 2
                    + (dst_pt[1] - transform_pt[1]) ** 2
                )
                if dist < threshold:
                    inliers.append(transform_pt)
                    src_inliers.append(src_pts[i])
                    dst_pts.remove(dst_pts[j])
                    break

        return inliers, src_inliers

    def get_sample_stars(self, sample_star, stars):
        """Returns 2 nearest neighbors of the given sample star"""
        temp_list = list(stars)
        temp_list.remove(sample_star)
        nn = self.find_nearest_neighbors(sample_star, temp_list)
        n1 = temp_list[nn[0]]
        n2 = temp_list[nn[1]]
        samples_stars = list()
        samples_stars.append(sample_star)
        samples_stars.append(n1)
        samples_stars.append(n2)
        return samples_stars

    def algorithm(self, stars1, stars2, num_iterations, threshold):
        """
        Get two sets of points as stars
        Make 1000 iterations:
        2.1. Get random star from the src set
        2.2. Get random star from the dst set
        2.3. Find the two nearest neighbors of every star and calculate the angles, a1 a2.
        2.4. If the absulote value of a1 - a2 is less that 4 then: make the tranform matrix from the six point of the images check inliners by using the matrix to get points from src to dst, with loop: if the transformed point is close to the dst point with a treshhold then add the transformed point to the inliners.
        2.5. If the new inliners set is bigger than the past one then the new inliners will be our best inliners.
        return inliners
        """
        inliers = []
        src_inliners = []
        src_stars = self.stars_list_to_array(stars1)
        dst_stars = self.stars_list_to_array(stars2)
        for i in range(num_iterations):
            src_sample = self.random_sample(stars=src_stars, num_samples=1)[0]
            src_samples_stars = self.get_sample_stars(
                sample_star=src_sample, stars=src_stars
            )
            angle = self.get_angle(src_samples_stars)
            dst_sample = self.random_sample(stars=dst_stars, num_samples=1)[0]
            dst_samples_stars = self.get_sample_stars(
                sample_star=dst_sample, stars=dst_stars
            )
            dst_angle = self.get_angle(dst_samples_stars)
            if abs(angle - dst_angle) < 5:
                matrix = self.make_transform(src_samples_stars, dst_samples_stars)
                crr_inliners, crr_src_inliners = self.check_inliers(
                    src_stars, dst_stars, matrix, threshold, stars1, stars2
                )
                if len(crr_inliners) >= len(inliers):
                    inliers = crr_inliners
                    src_inliners = crr_src_inliners

        return inliers, src_inliners

    def run_nanopi(self, image1, image2):
        """Run the algorithm on two images on nanppi board."""
        print("Running Algorithm")
        stars1 = self.detect_nanopi(img=image1)
        stars2 = self.detect_nanopi(img=image2)
        dst_inliner, src_inliners = self.algorithm(
            stars1=stars1, stars2=stars2, num_iterations=1000, threshold=22
        )
        self.draw_results(img=image1, stars=src_inliners, image_name="src.png")
        self.draw_results(img=image2, stars=dst_inliner, image_name="dst.png")
        return dst_inliner, src_inliners
