import scipy.spatial.distance as scipy_dist
import numpy as np


def dbscan(data, eps, min_pts):
    dist_matrix = scipy_dist.squareform(scipy_dist.pdist(data))
    neighorbood = []

    for i in range(data.shape[0]):
        neighorbood.append([j for j in range(data.shape[0])
                            if dist_matrix[i][j] <= eps])

    core_points = [i for i, pt in enumerate(neighorbood) if len(pt) >= min_pts]

    border_point_candidates = list(set([i for i in range(len(neighorbood))])
                                   .difference(set(core_points)))

    border_point = [candidate for candidate in border_point_candidates
                    if any(
                        [pt in neighorbood[candidate] for pt in core_points]
                    )]

    noise_points = [i for i in range(
        len(neighorbood)) if not i in core_points and not i in border_point]

    print(
        f'core points: {list(map(lambda p: p + 1, core_points))} \nborder points: {list(map(lambda p: p + 1, border_point))} \nnoise points: {list(map(lambda p: p + 1, noise_points))}')

    clusters = []
    not_allocated = []
    has_visited = [False for i in range(len(neighorbood))]

    def get_all_density_reachable(pt, neighorbood, has_visited):
        if has_visited[pt]:
            return set()

        has_visited[pt] = True
        reachable_pts = set([pt])

        for i in neighorbood[pt]:
            if i == pt:
                continue
            reachable_pts = reachable_pts.union(get_all_density_reachable(
                i, neighorbood, has_visited))

        return reachable_pts

    for pt in range(len(neighorbood)):
        if(has_visited[pt]):
            continue

        if(pt in core_points):
            pts = list(get_all_density_reachable(pt, neighorbood, has_visited))
            clusters.append(pts)
        elif(pt in border_point):
            continue
        else:
            not_allocated.append(pt)

    for i in range(len(clusters)):
        print(f'C{i + 1}: {list(map(lambda p: p + 1, clusters[i]))}')

    print(f'Not in cluster: {list(map(lambda p: p + 1, not_allocated))}')


if __name__ == '__main__':
    data = np.array([[0, 0], [1, 0], [1, 1], [2, 2], [
                    3, 1], [3, 0], [0, 1], [3, 2], [6, 3]])
    dbscan(data, 1, 3)
