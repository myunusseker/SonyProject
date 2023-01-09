import numpy as np
import open3d as o3d


#table: 0.4 > point[0] > -0.35 and 0.05 > point[1] > -0.5 and 0.88 > point[2] > 0.8:
def extract_object_from_boundaries(point_cloud, x0, x1, y0, y1, z0, z1, invert_z=True):
    pcd = o3d.geometry.PointCloud(point_cloud)
    points = []
    rgbs = []
    for point, rgb in zip(np.asarray(pcd.points), np.asarray(pcd.colors)):
        if x0 > point[0] > x1 and y0 > point[1] > y1 and z0 > point[2] > z1:
            if invert_z:
                point[2] *= -1
            points.append(point)
            rgbs.append(rgb)

    points = np.stack(points)
    rgbs = np.stack(rgbs)
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(rgbs)
    pcd.estimate_normals(fast_normal_computation=True)
    pcd.orient_normals_to_align_with_direction()

    return pcd

def extract_plane(point_cloud):
    pcd = o3d.geometry.PointCloud(point_cloud)
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)

    plane_cloud = pcd.select_by_index(inliers)
    plane_cloud.paint_uniform_color([1.0, 0, 0])

    noneplane_cloud = pcd.select_by_index(inliers, invert=True)
    noneplane_cloud.paint_uniform_color([0, 0, 1.0])

    return plane_cloud, noneplane_cloud

def fill_bottom_convex_hull(point_cloud):
    pcd = o3d.geometry.PointCloud(point_cloud)
    bbox = pcd.get_axis_aligned_bounding_box()
    b_min, b_max = bbox.min_bound, bbox.max_bound

    points = []
    rgbs = []
    normals = []

    for point in np.asarray(pcd.points):
        points.append([point[0], point[1], b_min[2]])
        rgbs.append([1,0,0])
        normals.append([0,0,-1])

    points = np.stack(points)
    rgbs = np.stack(rgbs)
    normals = np.stack(normals)
    pcd.points = o3d.utility.Vector3dVector(np.concatenate((np.asarray(pcd.points), points), axis=0))
    pcd.colors = o3d.utility.Vector3dVector(np.concatenate((np.asarray(pcd.colors), rgbs), axis=0))
    pcd.normals = o3d.utility.Vector3dVector(np.concatenate((np.asarray(pcd.normals), normals), axis=0))
    
    #o3d.visualization.draw_geometries([pcd], point_show_normal=True)
    
    #pcd.estimate_normals(fast_normal_computation=True)
    #o3d.visualization.draw_geometries([pcd], point_show_normal=True)
    #pcd.orient_normals_to_align_with_direction()
    #pcd.orient_normals_consistent_tangent_plane(100)
    
    return pcd

if __name__=="__main__":
    pcd = o3d.io.read_point_cloud("data/1667251288885270.pcd", remove_nan_points=True)

    object_pcd = extract_object_from_boundaries(pcd, 0.14, 0.015, -0.055, -0.175, 0.88, 0.8)
    
    #o3d.visualization.draw_geometries([object_pcd], point_show_normal=True)
    
    object_pcd = fill_bottom_convex_hull(object_pcd)

    #o3d.visualization.draw_geometries([object_pcd], point_show_normal=True)

    poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(object_pcd, depth=10, width=0, scale=2, linear_fit=False)[0] 

    
    poisson_mesh.remove_degenerate_triangles()
    poisson_mesh.remove_duplicated_triangles()
    poisson_mesh.remove_duplicated_vertices()
    poisson_mesh.remove_non_manifold_edges()
    poisson_mesh.remove_unreferenced_vertices()
    #o3d.visualization.draw_geometries([poisson_mesh])
    #o3d.io.write_triangle_mesh("lego.obj", poisson_mesh)

    poisson_mesh.compute_vertex_normals()
    
    poisson_mesh.remove_degenerate_triangles()
    poisson_mesh.remove_duplicated_triangles()
    poisson_mesh.remove_duplicated_vertices()
    poisson_mesh.remove_non_manifold_edges()
    poisson_mesh.remove_unreferenced_vertices()

    o3d.visualization.draw_geometries([poisson_mesh])
    mesh = poisson_mesh.simplify_quadric_decimation(target_number_of_triangles=300)
    o3d.visualization.draw_geometries([mesh])
    o3d.io.write_triangle_mesh("lego_300.obj", mesh)

    voxel_size = max(poisson_mesh.get_max_bound() - poisson_mesh.get_min_bound()) / 9
    print(f'voxel_size = {voxel_size:e}')
    mesh_smp = poisson_mesh.simplify_vertex_clustering(voxel_size=voxel_size, contraction=o3d.geometry.SimplificationContraction.Average)
    print(f'Simplified mesh has {len(mesh_smp.vertices)} vertices and {len(mesh_smp.triangles)} triangles')
    o3d.visualization.draw_geometries([mesh_smp])
    o3d.io.write_triangle_mesh("lego_9.obj", mesh_smp)
