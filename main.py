import open3d as o3d
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
from object2urdf import ObjectUrdfBuilder
import pybullet as p
import time
import pybullet_data


def filter_white(pcd, tolerance=1.55):
    # Extract the points and colors
    p = pcd.points
    c = pcd.colors

    # Define a function to check if a color is similar to white
    def is_white(color):
        return color[0] + color[1] + color[2] >= tolerance

    # Use a list comprehension to get the indices of the points that are similar to white
    indices = [i for i in range(len(c)) if is_white(c[i])]

    # Filter out the points that are similar to white
    p = np.delete(p, indices, axis=0)
    c = np.delete(c, indices, axis=0)

    # Create a new point cloud object from the filtered points and colors
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p))
    pcd.colors = o3d.utility.Vector3dVector(c)

    return pcd


def filter_table(pcd, table=(-0.3, -0.5, 0.7, 0.35, 0.0, 0.91)):
    min_x, min_y, min_z, max_x, max_y, max_z = table
    p = np.asarray(pcd.points)
    c = np.asarray(pcd.colors)

    filtered_pc = [([p[i][0], p[i][1], -p[i][2]], c[i]) for i in range(len(p)) if
                   min_x <= p[i][0] <= max_x and min_y <= p[i][1] <= max_y and min_z <= p[i][2] <= max_z]

    p, c = zip(*filtered_pc)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p))
    pcd.colors = o3d.utility.Vector3dVector(c)
    return pcd


def load_pointcloud(filename):
    pcd = o3d.io.read_point_cloud(f"data/{filename}", remove_nan_points=True)

    pcd = filter_table(pcd)
    original_pcd = o3d.geometry.PointCloud(pcd)

    pcd = filter_white(pcd)

    return pcd, original_pcd


def find_convex_hull(pcd):
    # Extract the XY coordinates of the points
    points = np.asarray(pcd.points)

    # Find the convex hull of the points
    hull = ConvexHull(points[:, :2])

    hull_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[hull.vertices]))

    return hull_pcd


def segment_objects(pcd):
    # Extract the XYZ coordinates of the points
    points = np.asarray(pcd.points)

    # Use DBSCAN to cluster the points
    db = DBSCAN(eps=0.02, min_samples=100).fit(points)
    labels = db.labels_

    # Define an array of colors to use for the clusters
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]]

    # Create a point cloud for each cluster
    clusters = []
    hulls = []
    for i in range(max(labels) + 1):
        if np.sum(labels == i) < 1000:
            continue
        cluster = o3d.geometry.PointCloud()
        cluster.points = o3d.utility.Vector3dVector(points[labels == i])
        cluster.paint_uniform_color(colors[i % len(colors)])
        clusters.append(cluster)

        hulls.append(find_convex_hull(cluster))

    return clusters, hulls


def in_hull(points, hull):
    from scipy.spatial import Delaunay
    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)

    return hull.find_simplex(points) >= 0


def filter_objects(pcd, hulls):
    objects = []
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    for hull in hulls:
        hull_points = np.asarray(hull.points)[:, :2]
        obj = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[in_hull(points[:, :2], hull_points)]))
        obj.colors = o3d.utility.Vector3dVector(colors[in_hull(points[:, :2], hull_points)])
        obj.estimate_normals(fast_normal_computation=True)
        obj.orient_normals_to_align_with_direction()
        objects.append(obj)
    return objects


def fill_bottom_convex_hull(objects):
    result = []
    colors = []
    for obj in objects:
        pcd = o3d.geometry.PointCloud(obj)
        bbox = pcd.get_axis_aligned_bounding_box()
        b_min, b_max = bbox.min_bound, bbox.max_bound

        points = []
        rgbs = []
        normals = []
        color = np.asarray(pcd.colors).mean(axis=0)

        for point in np.asarray(pcd.points):
            points.append([point[0], point[1], b_min[2]])
            rgbs.append(color)
            normals.append([0, 0, -1])

        points = np.stack(points)
        rgbs = np.stack(rgbs)
        normals = np.stack(normals)
        pcd.points = o3d.utility.Vector3dVector(np.concatenate((np.asarray(pcd.points), points), axis=0))
        pcd.colors = o3d.utility.Vector3dVector(np.concatenate((np.asarray(pcd.colors), rgbs), axis=0))
        pcd.normals = o3d.utility.Vector3dVector(np.concatenate((np.asarray(pcd.normals), normals), axis=0))

        result.append(pcd)
        colors.append(color)

    return result, colors


def generate_meshes(objects):
    meshes = []
    for obj in objects:
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            obj,
            depth=8,
            width=0,
            scale=2,
            linear_fit=False
        )[0]

        mesh.compute_vertex_normals()
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
        mesh.remove_unreferenced_vertices()
        meshes.append(mesh)

    return meshes


def simplify_meshes(meshes):
    simple_meshes = []
    centers = []
    for i, mesh in enumerate(meshes):
        simple_mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=500)
        center = simple_mesh.get_center()
        """
        v_size = max(mesh.get_max_bound() - mesh.get_min_bound()) / 9
        simple_mesh = mesh.simplify_vertex_clustering(
            voxel_size=v_size,
            contraction=o3d.geometry.SimplificationContraction.Average
        )
        """
        simple_meshes.append(simple_mesh)
        centers.append([center[0], center[1], -center[2]])
        o3d.io.write_triangle_mesh(f"objects/object_{i}.obj", simple_mesh)
    return simple_meshes, centers


def create_urdfs(N):
    for i in range(N):
        builder = ObjectUrdfBuilder('./', urdf_prototype='objects/_prototype.urdf')
        builder.build_urdf(
            filename=f"objects/object_{i}.obj",
            force_overwrite=True,
            decompose_concave=True,
            force_decompose=False,
            center='mass'
        )


def create_sim(N, centers, colors):
    physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
    p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)
    p.setGravity(0, 0, -10)
    p.loadURDF("plane.urdf")
    startOrientation = p.getQuaternionFromEuler([0, 0, 0])
    tableID = p.loadURDF("objects/table.obj.urdf", basePosition=[0, 0, 1.33],
               baseOrientation=p.getQuaternionFromEuler([np.pi / 2, 0, 0]))
    p.changeVisualShape(tableID, -1, rgbaColor=[0.98, 0.83, 0.6, 1])
    #plateID = p.loadURDF("objects/plate.obj.urdf", basePosition=[0, -0.1, 1.68],
    #           baseOrientation=p.getQuaternionFromEuler([np.pi / 2, 0, 0]))
    #p.changeVisualShape(plateID, -1, rgbaColor=[1, 0.975, 0.95, 1])
    for i in range(N):
        boxId = p.loadURDF(f"objects/object_{i}.obj.urdf", [centers[i][0], centers[i][1], centers[i][2] + 1])
        p.changeVisualShape(boxId, -1, rgbaColor=[colors[i][0], colors[i][1], colors[i][2], 1])

    for i in range(10000):
        p.stepSimulation()
        time.sleep(1. / 480.)

    p.disconnect()


if __name__ == '__main__':
    pcd, original_pcd = load_pointcloud("scene1.pcd")

    o3d.visualization.draw_geometries([pcd])

    clusters, hulls = segment_objects(pcd)

    o3d.visualization.draw_geometries(clusters)

    objects = filter_objects(original_pcd, hulls)

    objects, colors = fill_bottom_convex_hull(objects)

    meshes = generate_meshes(objects)

    meshes, centers = simplify_meshes(meshes)

    o3d.visualization.draw_geometries(meshes)

    create_urdfs(len(meshes))

    create_sim(len(meshes), centers, colors)
