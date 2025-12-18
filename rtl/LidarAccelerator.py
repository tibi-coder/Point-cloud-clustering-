from pymtl3 import *


# Voxelizer (spatial discretization)
class Voxelizer(Component):
    def construct(s, nbits=16, voxel_bits=4):
        """
        voxel_bits: number of bits removed from precision.
        """
        s.in_x = InPort(mk_bits(nbits))
        s.in_y = InPort(mk_bits(nbits))
        s.in_z = InPort(mk_bits(nbits))

        s.out_x = OutPort(mk_bits(nbits))
        s.out_y = OutPort(mk_bits(nbits))
        s.out_z = OutPort(mk_bits(nbits))

        s.shift_amt = voxel_bits

        @update
        def comb_logic():
            # Snap coordinates to voxel grid
            s.out_x @= (s.in_x >> s.shift_amt) << s.shift_amt
            s.out_y @= (s.in_y >> s.shift_amt) << s.shift_amt
            s.out_z @= (s.in_z >> s.shift_amt) << s.shift_amt


# 3D Euclidean Squared Distance
class EuclideanDistSq3D(Component):
    def construct(s, nbits=16):
        s.x1 = InPort(mk_bits(nbits))
        s.y1 = InPort(mk_bits(nbits))
        s.z1 = InPort(mk_bits(nbits))

        s.x2 = InPort(mk_bits(nbits))
        s.y2 = InPort(mk_bits(nbits))
        s.z2 = InPort(mk_bits(nbits))

        # Wider output to avoid overflow
        WideType = mk_bits(nbits * 2 + 2)
        s.out_dist_sq = OutPort(WideType)

        @update
        def calc_logic():
            # Coordinate differences
            dx = s.x1 - s.x2
            dy = s.y1 - s.y2
            dz = s.z1 - s.z2

            # Sign extend before multiplication
            dx_w = sext(dx, WideType)
            dy_w = sext(dy, WideType)
            dz_w = sext(dz, WideType)

            # Squared Euclidean distance
            s.out_dist_sq @= (dx_w * dx_w) + (dy_w * dy_w) + (dz_w * dz_w)


#  LiDAR processing core
class LidarCore(Component):
    def construct(s, nbits=16):
        # Raw input point
        s.in_x = InPort(mk_bits(nbits))
        s.in_y = InPort(mk_bits(nbits))
        s.in_z = InPort(mk_bits(nbits))

        # Floor threshold
        s.floor_thresh = InPort(mk_bits(nbits))

        # Cluster centroids (K = 2)
        s.c1_x, s.c1_y, s.c1_z = [InPort(mk_bits(nbits)) for _ in range(3)]
        s.c2_x, s.c2_y, s.c2_z = [InPort(mk_bits(nbits)) for _ in range(3)]

        # Outputs
        s.out_cluster = OutPort(mk_bits(1))  # 0 or 1
        s.out_valid = OutPort(mk_bits(1))    # 1 = valid point

        # Instantiate voxelizer
        s.voxelizer = Voxelizer(nbits, voxel_bits=4)

        # distance units
        s.dist1 = EuclideanDistSq3D(nbits)
        s.dist2 = EuclideanDistSq3D(nbits)

        # Connections

        # Input -> voxelizer
        s.voxelizer.in_x //= s.in_x
        s.voxelizer.in_y //= s.in_y
        s.voxelizer.in_z //= s.in_z

        # Voxelized point -> distance unit 1
        s.dist1.x1 //= s.voxelizer.out_x
        s.dist1.y1 //= s.voxelizer.out_y
        s.dist1.z1 //= s.voxelizer.out_z

        # Centroid 1
        s.dist1.x2 //= s.c1_x
        s.dist1.y2 //= s.c1_y
        s.dist1.z2 //= s.c1_z

        # Voxelized point -> distance unit 2
        s.dist2.x1 //= s.voxelizer.out_x
        s.dist2.y1 //= s.voxelizer.out_y
        s.dist2.z1 //= s.voxelizer.out_z

        # Centroid 2
        s.dist2.x2 //= s.c2_x
        s.dist2.y2 //= s.c2_y
        s.dist2.z2 //= s.c2_z

        @update
        def control_logic():
            # Floor removal using voxelized Z
            if s.voxelizer.out_z < s.floor_thresh:
                s.out_valid @= 0
            else:
                s.out_valid @= 1

            # Choose closest centroid
            if s.dist1.out_dist_sq < s.dist2.out_dist_sq:
                s.out_cluster @= 0
            else:
                s.out_cluster @= 1
