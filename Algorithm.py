from Star import *
import cv2
import numpy as np
from scipy.spatial import KDTree
import math
from PIL import Image


class Algorithm:
    def __init__(self):
        self.min_area = 1
        self.max_area = 1000

    def find_nearest_neighbors(self, query_point, points, k=2):
        kdtree = KDTree(points)
        distances, indices = kdtree.query(query_point, k=k)
        return indices.tolist()

    def get_angle(self, points):
        p1 = math.dist(points[0], points[1])
        p2 = math.dist(points[1], points[2])
        p3 = math.dist(points[2], points[0])
        angle = math.acos((p1**2 + p2**2 - p3**2) / (2 * p1 * p2))
        return math.degrees(angle)

    def draw_results(self, img, stars, image_name):
        print("Drawing results")
        image = cv2.imread(img)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(image_name, gray)
        image = np.array(Image.open(image_name).resize((600, 600)))
        radius = 20
        for i in range(len(stars)):
            cv2.circle(image, (int(stars[i][0]), int(
                stars[i][1])), radius, (255, 255, 255), 2)
            cv2.putText(image, str(i), (int(stars[i][0]-10), int(
                stars[i][1])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imwrite(image_name, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def detect(self, image):
        img = cv2.imread(str(image))
        img = np.array(Image.open(image).resize((600, 600)))
        if len(img.shape) == 2:
            gray = img
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, 1, 20, param1=240, param2=0.5, minRadius=2, maxRadius=7)
        stars = []
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(img, (x, y), r+10, (255, 0, 0), 2)
            star_mask = np.zeros_like(gray)
            cv2.circle(star_mask, (x, y), r, 255, -1)
            brightness = cv2.mean(gray, mask=star_mask)[0]
            stars.append(Star(
                x=x, y=y, brightness=brightness, radius=r))
        print(len(stars))

        cv2.imwrite(image+"stars.png", img)
        return stars

    def stars_list_to_array(self, stars):
        list = []
        for star in stars:
            list.append((star.x, star.y))
        return list

    def random_sample(self, stars, num_samples):
        index = np.random.choice(
            len(stars), num_samples, replace=False)
        return [stars[i] for i in index]

    def make_transform(self, src_stars, dst_stars):
        src_pts = np.float32(src_stars)
        dst_pts = np.float32(dst_stars)
        matrix = cv2.getAffineTransform(src_pts, dst_pts)
        return matrix

    def check_inliers(self, src_pts_original, dst_pts_original, matrix, threshold, stars1, stars2):
        inliers = []
        src_inliers = []
        src_pts = list(src_pts_original)
        dst_pts = list(dst_pts_original)
        len1 = len(src_pts)
        len2 = len(dst_pts)
        for i in range(len1):
            src_pt = np.array(
                [src_pts[i][0], src_pts[i][1], 1], dtype=np.float32)
            len2 = len(dst_pts)
            for j in range(len2):
                dst_pt = np.array(
                    [dst_pts[j][0], dst_pts[j][1], 1], dtype=np.float32)
                transform_pt = np.dot(matrix, src_pt)
                dist = np.sqrt((dst_pt[0] - transform_pt[0])**2 + (
                    dst_pt[1] - transform_pt[1])**2)
                r1 = stars1[i].radius
                r2 = stars2[j].radius
                if dist < threshold and abs(r1 - r2) < 0.1:
                    inliers.append(transform_pt)
                    src_inliers.append(src_pts[i])
                    dst_pts.remove(dst_pts[j])
                    break

        return inliers, src_inliers

    def get_sample_stars(self, sample_star, stars):
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
        inliers = []
        src_inliners = []
        src_stars = self.stars_list_to_array(stars1)
        dst_stars = self.stars_list_to_array(stars2)
        for i in range(num_iterations):
            src_sample = self.random_sample(stars=src_stars, num_samples=1)[0]
            src_samples_stars = self.get_sample_stars(
                sample_star=src_sample, stars=src_stars)
            angle = self.get_angle(src_samples_stars)
            dst_sample = self.random_sample(stars=dst_stars, num_samples=1)[0]
            dst_samples_stars = self.get_sample_stars(
                sample_star=dst_sample, stars=dst_stars)
            dst_angle = self.get_angle(dst_samples_stars)
            if abs(angle - dst_angle) < 5:
                matrix = self.make_transform(
                    src_samples_stars, dst_samples_stars)
                crr_inliners, crr_src_inliners = self.check_inliers(
                    src_stars, dst_stars, matrix, threshold, stars1, stars2)
                if len(crr_inliners) >= len(inliers):
                    inliers = crr_inliners
                    src_inliners = crr_src_inliners

        return inliers, src_inliners

    def run(self, image1, image2):
        print("Running Algorithm")
        stars1 = self.detect(image=image1)
        stars2 = self.detect(image=image2)
        inliner, src_inliners = self.algorithm(stars1=stars1, stars2=stars2,
                                               num_iterations=1000, threshold=22)
        self.draw_results(
            img=image1, stars=src_inliners, image_name="src.png")
        self.draw_results(
            img=image2, stars=inliner, image_name="dst.png")
        return inliner, src_inliners
