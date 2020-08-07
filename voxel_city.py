# This script randomly generates a colorful voxel city and writes it to a
# MagicaVoxel .vox file (so MagicaVoxel can render it). See the main block at
# the bottom of the file for configuration variables.
#
# Algorithm overview:
# - Generate a floorplan for the city blocks, which are neatly aligned along the
#   cities axes.
# - Fill in each city block with buildings.
#
# Two files are produced. One is the 'city.vox' file which contains the city
# voxel scene. A 'city.png' file is also produced, which is the city's
# floorplan.


import struct
import time
from PIL    import (Image, ImageDraw)
from random import (seed, randint, shuffle)
from typing import (List, Tuple, Dict, Set, Optional)
from math   import (floor, ceil)


################################################################################
# # # # # # # # # # # # # # # # CITY GENERATION  # # # # # # # # # # # # # # # #
################################################################################


# Colors for the building's windows. These are given assigned arbitrary constant
# values, which are allocated in the voxel color palette (at index 0,1,2).
# These should be manually tweaked in MagicaVoxel before rendering.
WINDOW_COLOR1 = (255,255,255,255)
WINDOW_COLOR2 = (254,254,254,255)
WINDOW_COLOR3 = (253,253,253,255)


def generate_bins( size: int, min_bin_size: int, max_bin_size: int, spacing: int ) -> Optional[List[int]]:
  """Generates bins of random size - between the given bounds - which fill the
  entire given size when intercalated with spacing. This is used to generate the
  large city blocks (along a single axis).

  Examples
  --------
  >>> generate_bins(20, 3, 5, 1)
  [3, 5, 4, 5]

  Note that 3+1+5+1+4+1+5 = 20

  Returns
  -------
  list, optional
    The sizes of the fitted bins. If surely no bins fit, this is 'None'.
  """
  if size == 0:
    return []
  elif size < min_bin_size: # No bin will ever fit
    return None
  elif size < spacing + min_bin_size * 2:
    # There'll surely not fit two bins anymore
    if size <= max_bin_size:
      return [size] # Just fit the last one in
    else:
      return None # Try again
  else: # At least 2 bins will fit
    max_bin_size = min( max_bin_size, size - spacing - min_bin_size )
    
    # Choose a size for the current bin, and recurse. If no bins are found in
    # the recursive case, it will try another bin size and try again. This
    # repeats until an appropriate bin is found, or all are tried.
    # Note that this potentially causes exponential-time execution. This should
    # not be a problem when reasonable parameters are chosen.
    possible_bin_sizes = list( range( min_bin_size, max_bin_size + 1 ) )
    shuffle( possible_bin_sizes )
    for bin_size in possible_bin_sizes:
      next_bins = generate_bins( size - bin_size - spacing, min_bin_size, max_bin_size, spacing )
      if next_bins is not None:
        return [ bin_size ] + next_bins
    return None


def generate_buildings( width: int, depth: int, min_bin: int, max_bin: int, spacing: int, x_addend: int = 0, y_addend: int = 0 ) -> Optional[List[Tuple[int,int,int,int]]]:
  """Generates the floor plan for the buildings within the given city block
  rectangle. It repeatedly splits at a random point along either axis
  (similar to a k-d tree), which produces a random-appearing distribution of
  buildings.

  Parameters
  ----------
  width : int
    Width of the city block
  depth : int
    Depth of the city block
  min_bin : int
    Minimum size of a building along either axis
  max_bin : int
    Maximum size of a building along either axis
  spacing : int
    The spacing between the buildings - this is used for roads
  x_addend : int
    Accumulator for recursion. Defines an x-offset that should added to the
    building's x-bounds.
  y_addend : int
    Accumulator for recursion. Defines an y-offset that should added to the
    building's y-bounds.
  
  Returns
  -------
  list, optional
    List of bounds (x,y,width,depth) for each building. If surely no bins fit,
    this is 'None'.
  """
  if width < min_bin or depth < min_bin:
    return None

  can_split_width = ( width >= min_bin * 2 + spacing )
  can_split_depth = ( depth >= min_bin * 2 + spacing )

  if not can_split_width and not can_split_depth:
    if width <= max_bin and depth <= max_bin:
      return [(x_addend, y_addend, width, depth)]
    else:
      return None
  else:
    # A split is possible along at least either of the axes. If there is only
    # one possible axis, choose that one. Otherwise pick randomly.
    should_split_width = not can_split_depth or ( can_split_width and randint( 0, 1 ) == 1 )
    
    # For the chosen axis, keep randomly trying splits, until one is found that
    # completely fills the space with buildings. Or return 'None' if it does not
    # exist. Note that this could cause exponential-time execution, but this
    # does not happen when reasonable parameters are chosen (see main).
    if should_split_width: # Split width
      possible_split_xs = list( range( min_bin, width - min_bin - spacing + 1 ) )
      shuffle( possible_split_xs )

      for split_x in possible_split_xs:
        res1 = generate_buildings( split_x, depth, min_bin, max_bin, spacing, x_addend, y_addend )
        res2 = generate_buildings( width - split_x - spacing, depth, min_bin, max_bin, spacing, x_addend + split_x + spacing, y_addend )

        if res1 is not None and res2 is not None:
          return res1 + res2

      return None # No fitting buildings were found
    else: # Split depth
      possible_split_ys = list( range( min_bin, depth - min_bin - spacing + 1 ) )
      shuffle( possible_split_ys )

      for split_y in possible_split_ys:
        res1 = generate_buildings( width, split_y, min_bin, max_bin, spacing, x_addend, y_addend )
        res2 = generate_buildings( width, depth - split_y - spacing, min_bin, max_bin, spacing, x_addend, y_addend + split_y + spacing )

        if res1 is not None and res2 is not None:
          return res1 + res2
          
      return None # No fitting buildings were found


def randodd( l: int, h: int ) -> int:
  """Returns a random odd number within the given range (both inclusive). This
  raises a ValueError when there is no value in the given range.
  """
  return 2 * randint( ceil( ( l - 1 ) / 2 ), floor( ( h - 1 ) / 2 ) ) + 1


def randlowch( ) -> int:
  """Randomly returns any value from [0,64,128]. Used for the lower-saturated
  color channels. Note that these have a limited domain to ensure no more than
  255 unique colors are present in the palette.
  """
  return randint(0,2)*64


def randhighch( ) -> int:
  """Randomly returns any value from [128,192,255]. Used for the
  higher-saturated color channels. Note that these have a limited domain to
  ensure no more than 255 unique colors are present in the palette.
  """
  return min(randint(2,4)*64, 255)


def rand_building_color( ) -> Tuple[int,int,int,int]:
  """Randomly generates a color with at least one highly-saturated color
  channel. These are used for the building exteriors.
  """
  choice = randint(0,5)
  if choice == 0: # red (0xff0000)
    return ( randhighch(),  randlowch(),  randlowch(), 255 )
  elif choice == 1: # green (0x00ff00)
    return (  randlowch(), randhighch(),  randlowch(), 255 )
  elif choice == 2: # blue (0x0000ff)
    return (  randlowch(),  randlowch(), randhighch(), 255 )
  elif choice == 3: # pink (0xff00ff)
    return ( randhighch(),  randlowch(), randhighch(), 255 )
  elif choice == 4: # cyan (0x00ffff)
    return (  randlowch(), randhighch(), randhighch(), 255 )
  else: # yellow (0xffff00)
    return ( randhighch(), randhighch(),  randlowch(), 255 )


def rand_window_color( ) -> Optional[Tuple[int,int,int,int]]:
  """Generates a color for a building's window. These colors are abitrary and
  should be manually tweaked in MagicaVoxel before rendering. When 'None' is
  returned, there should be no window.
  """
  choice = randint(0,5)
  if choice <= 2:
    return None
  elif choice == 3:
    return WINDOW_COLOR1
  elif choice == 4:
    return WINDOW_COLOR2
  else:
    return WINDOW_COLOR3


def nextpow2( x: int ) -> int:
  """Returns the lowest power-of-2 integer that is at least as large as 'x'."""
  v = 1
  while v < x:
    v = v * 2
  return v


class VoxelBlock:
  """A block of voxels. It's objective is easy extraction of individual (sparse)
  voxels. Voxels are represented by a single integer, which represents their
  index in the color palette (See 'VoxelBuilder').
  """

  def __init__( self, x_size: int, y_size: int, z_size: int ):
    assert( x_size > 0 and y_size > 0 and z_size > 0)

    self.x_size = x_size
    self.y_size = y_size
    self.z_size = z_size
    # Maps voxel locations to indices in the color palette.
    # 'self.data[(x,y,z)] = color_i'
    self._data: Dict[Tuple[int,int,int],int] = {}
  
  def set( self, x: int, y: int, z: int, color_i: int ):
    assert( x >= 0 and y >= 0 and z >= 0 and x < self.x_size and y < self.y_size and z < self.z_size )

    self._data[(x,y,z)] = color_i
  
  def voxels( self ) -> List[Tuple[int,int,int,int]]:
    return [(x,y,z,i) for (x,y,z),i in self._data.items()]


class VoxelBuilder:
  """Builds a voxel scene. A scene consists of several voxel blocks (as
  MagicaVoxel limits block sizes to 126x126x126) with a color palette. Yet, the
  interface provided by this class abstracts away from that, and can be modified
  as one a single voxel mesh.
  """

  def __init__( self, x_size: int, y_size: int, z_size: int ):
    self.x_size = x_size
    self.y_size = y_size
    self.z_size = z_size

    # The palette is in inverse map from (r,g,b,a) to their index. When colors
    # are added to the palette, they are giving the next-available index.
    self._palette: Dict[Tuple[int,int,int,int],int] = {}

    num_x_blocks = ceil( self.x_size / 126 )
    num_y_blocks = ceil( self.y_size / 126 )
    num_z_blocks = ceil( self.z_size / 126 )
    
    # Note that a scene consists of several voxel blocks that are at most
    # 126x126x126 voxels in size (mandated by MagicaVoxel).
    self._blocks: List[List[List[VoxelBlock]]] = []

    for block_z in range(0,num_z_blocks):
      plane = []
      for block_y in range(0,num_y_blocks):
        line = []
        for block_x in range(0,num_x_blocks):
          block_x_size = min( 126, self.x_size - 126 * block_x )
          block_y_size = min( 126, self.y_size - 126 * block_y )
          block_z_size = min( 126, self.z_size - 126 * block_z )
          line.append( VoxelBlock( block_x_size, block_y_size, block_z_size ) )
        plane.append( line )
      self._blocks.append( plane )
  

  def set( self, x: int, y: int, z: int, x_size: int, y_size: int, z_size: int, color: Tuple[int,int,int,int] ):
    """Sets all voxels in the specified region to the given color"""

    assert( x >= 0 and y >= 0 and z >= 0 and x + x_size <= self.x_size and y + y_size <= self.y_size and z + z_size <= self.z_size )

    color_i = self.add_to_palette( color )

    for iz in range(z,z+z_size):
      for iy in range(y,y+y_size):
        for ix in range(x,x+x_size):
          block_x = ix // 126
          block_y = iy // 126
          block_z = iz // 126

          block = self._blocks[ block_z ][ block_y ][ block_x ]

          block.set( ix % 126, iy % 126, iz % 126, color_i )


  def add_to_palette( self, color: Tuple[int,int,int,int] ) -> int:
    """Adds a color to the color palette and returns its index. If it already
    exists, the current index is returned. Note that it is not allowed to have
    more than 255 distinct colors.
    """
    if color in self._palette:
      return self._palette[ color ]
    else:
      color_i = len( self._palette )
      if color_i > 254:
        # If the new index is larger than 254, it means there are at least 256
        # colors, which is not allowed
        raise Exception( 'TooManyColors' )
      self._palette[ color ] = color_i
      return color_i
  

  def to_models( self ) -> List[ Tuple[ Tuple[int,int,int], Tuple[int,int,int], List[Tuple[int,int,int,int]] ] ]:
    """Converts the entire voxel mesh into a scene. A scene may consist of
    multiple voxel blocks that are each at most 126x126x126 in size.

    Returns
    -------
    list
      A list of voxel blocks. Every block consists of:
      * (x,y,z) - location of the block in the world
      * (x_size,y_size,z_size) - size of the block
      * [(x,y,z,i)] - Voxels in the mesh. x,y,z are relative to the block.
          'i' references a color in the palette.
    """
    models = []

    num_x_blocks = ceil( self.x_size / 126 )
    num_y_blocks = ceil( self.y_size / 126 )
    num_z_blocks = ceil( self.z_size / 126 )

    for block_z in range(0, num_z_blocks):
      for block_y in range(0, num_y_blocks):
        for block_x in range(0, num_x_blocks):
          block_x_size = min( 126, self.x_size - 126 * block_x )
          block_y_size = min( 126, self.y_size - 126 * block_y )
          block_z_size = min( 126, self.z_size - 126 * block_z )

          block = self._blocks[ block_z ][ block_y ][ block_x ]

          if block is not None:
            voxels = block.voxels( )
            
            if len( voxels ) > 0: # Don't store empty blocks
              # Note that blocks are translated by their center
              x = block_x*126 - 1000 + ceil(block_x_size/2)
              y = block_y*126 - 1000 + ceil(block_y_size/2)
              z = ceil(block_z_size/2)
              models.append( ( (x, y, z), (block_x_size, block_y_size, block_z_size), voxels ) )

    return models
  

  def to_vox( self ) -> bytes:
    """Converts the scene into a MagicaVoxel .vox format binary representation.
    See also the functions (prefixed with 'vox_') below.
    """
    w = bytearray( )
    w.extend( b'VOX ' )
    w.extend( struct.pack( '<I', 150 ) )

    # Store for each block (x,y,z), (xsize, ysize, zsize), [(vx,vy,vz,color_index)]
    models = self.to_models( )
    
    main_children_w = bytearray( )

    # While the specification requires a 'pack' chunk, it is not present in any
    # .vox files produced by MagicaVoxel. Rather, including it causes
    # MagicaVoxel not to recognise the file at all. So, leave it out.
    #
    # if len( models ) != 1:
    #   main_children_w.extend( vox_chunk( b'PACK', b'', vox_pack_content( len( models ) ) ) )

    # First, the models are included in the file. Model indices refer to the
    # models in the order they appear in the file. (i.e. the first model has
    # index 0)
    for ((_x,_y,_z), (x_size,y_size,z_size), voxels) in models:
      main_children_w.extend( vox_chunk( b'SIZE', vox_size_content( x_size, y_size, z_size ), b'' ) )
      main_children_w.extend( vox_chunk( b'XYZI', vox_xyzi_content( voxels ), b'' ) )

    # The models are included in a scene graph, which describes where each
    # model appears in the scene. This graph is simply structured as follows:
    #
    #       T
    #       |
    #       G
    #   / |...|  \
    #  T  T...T   T
    #  |  |...|   |
    #  S  S...S   S
    # 
    # T's represent transform nodes. G's represent group nodes. S's represent
    # shape nodes, each of which references a model.

    # Node 0 is the root transform
    main_children_w.extend( vox_chunk( b'nTRN', vox_ntrn_content( 0, 1, -1, (0,0,0) ), b'' ) )
    # Nodes 1 is the root group
    main_children_w.extend( vox_chunk( b'nGRP', vox_ngrp_content( 1, list( range( 2, 2+2*len(models), 2 ) ) ), b'' ) )

    # Nodes '2+2*i' are the transform nodes. Nodes '2+2*i+1' are the shape nodes
    for i in range(0, len(models)):
      ((x,y,z), (x_size,y_size,z_size), _voxels) = models[ i ]
      main_children_w.extend( vox_chunk( b'nTRN', vox_ntrn_content( i*2+2, i*2+3, 0, (x,y,z) ), b'' ) )
      main_children_w.extend( vox_chunk( b'nSHP', vox_nshp_content( i*2+3, i ), b'' ) )

    main_children_w.extend( vox_chunk( b'RGBA', vox_rgba_content( self._palette ), b'' ) )

    w.extend( vox_chunk( b'MAIN', b'', bytes( main_children_w ) ) )

    return bytes( w )


################################################################################
# # # # # # # # # # # MAGICAVOXEL .VOX BINARY GENERATION # # # # # # # # # # # #
################################################################################

# These functions each produce part of the binary representation of a .vox file.
# The format specifications are taken from:
# * https://github.com/ephtracy/voxel-model/blob/master/MagicaVoxel-file-format-vox.txt
# * https://github.com/ephtracy/voxel-model/blob/master/MagicaVoxel-file-format-vox-extension.txt
#
# Note that this is not a full implementation of the specification. Only the
# necessary chunks and chunk-fields are implemented.


def vox_chunk( id: bytes, content: bytes, children: bytes ) -> bytes:
  """Produces the chunk structure. Every chunk has a 4-byte long identifier
  (e.g. 'SIZE'), optional content, and optional child chunks. The payload
  and child chunks should by constructed by external functions, this function
  packs them together.

  Example
  -------
  >>> vox_chunk( 'SIZE', vox_size_content(10,10,10), b'' )
  
  While 'vox_size_content(..)' produces the content for the chunk, the full
  chunk (without child chunks) composed by this function.
  """
  assert( len( id ) == 4 )
  w = bytearray( )
  w.extend( id )
  w.extend( struct.pack( '<II', len( content ), len( children ) ) )
  w.extend( content )
  w.extend( children )
  return bytes( w )


# Note that this function is *not* used. MagicaVoxel refuses to accept files
# that contain this chunk.
def vox_pack_content( num_models: int ) -> bytes:
  """Produces the content for the PACK chunk, which contains the model count."""
  return struct.pack( '<I', num_models )


def vox_size_content( x_size: int, y_size: int, z_size: int ) -> bytes:
  """Produces the content for the SIZE chunk. It contains the dimensions for a
  single model.
  """
  assert( x_size > 0 and y_size > 0 and z_size > 0 and x_size <= 126 and y_size <= 126 and z_size <= 126 )
  return struct.pack( '<III', x_size, y_size, z_size )


def vox_xyzi_content( voxels: List[ Tuple[int,int,int,int] ] ) -> bytes:
  """Produces the content for the XYZI chunk, which contains the voxels for a
  single model.

  Parameters
  ----------
  voxels : list
    A list of voxels, each represented as (x,y,z,i). (x,y,z) are the voxel's
    location within the chunk. 'i' is the index of the voxel's color in the
    palette.
  """
  w = bytearray( )
  w.extend( struct.pack( '<I', len( voxels ) ) )
  for (x,y,z,i) in voxels:
    assert( x < 126 and y < 126 and z < 126 and i <= 254 )
    # Add 1 to the index, as the specification requires that index 0 in the
    # palette (in memory) is skipped. So, the first color resides at index 1 in
    # the palette.
    w.extend( struct.pack( '<BBBB', x, y, z, i + 1 ) )
  return bytes( w )


def vox_rgba_content( palette: Dict[ Tuple[int,int,int,int], int ] ) -> bytes:
  """Produces the content for the RGBA chunk, which contains the color palette
  for the entire scene.
  """
  # While the palette contains at most 255 unique colors (indices 0-254), the
  # palette is stored as 256 entries. So, the last is stored as 0x00000000.
  assert( len( palette ) <= 255 )

  # The palette is stored as a Dict[(r,g,b,a),i]. Make this into an array such
  # that palette_data[i]=(r,g,b,a)
  palette_data: List[Tuple[int,int,int,int]] = [(0,0,0,0) for i in range(0,256)]
  for (r,g,b,a),i in palette.items():
    palette_data[i] = (r,g,b,a)

  w = bytearray( )
  for (r,g,b,a) in palette_data:
    w.extend( struct.pack( '<BBBB', r, g, b, a ) )
  return bytes( w )

def vox_ntrn_content( node_id: int, child_node_id: int, layer_id: int, translation: Tuple[int,int,int] ) -> bytes:
  """Produces the content for the nTRN chunk, which is a transform node in the
  scene graph.

  Parameters
  ----------
  node_id : int
    The unique identifier of this node. Used by other nodes to reference it
  child_node_id : int
    The identifier of the child node (which is transformed by this node)
  layer_id : int
    The layer within which this node resides. Not too important. However, the
    root-transform node should reside at layer -1 (0xFFFFFFFF); at least, this
    seems to be the case for .vox files produced by MagicaVoxel.
  translation : (int,int,int)
    The child chunk is translated by this amount
  """
  (x,y,z) = translation
  w = bytearray( )
  w.extend( struct.pack( '<I', node_id ) )
  w.extend( vox_dict( [ ] ) )
  w.extend( struct.pack( '<I', child_node_id ) )
  w.extend( struct.pack( '<i', -1 ) ) # reserved
  w.extend( struct.pack( '<i', layer_id ) ) # layer id
  w.extend( struct.pack( '<I', 1 ) ) # num frames

  dict_vals = []

  if x != 0 or y != 0 or z != 0:
    dict_vals.append( ( '_t', '{} {} {}'.format( x, y, z ) ) )

  w.extend( vox_dict( dict_vals ) )

  return bytes( w )


def vox_ngrp_content( node_id: int, child_node_ids: List[int] ) -> bytes:
  """Produces the content for the nGRP chunk, which is a group node in the scene
  graph.
  """
  w = bytearray( )
  w.extend( struct.pack( '<I', node_id ) )
  w.extend( vox_dict( [] ) )
  w.extend( struct.pack( '<I', len( child_node_ids ) ) )
  for child_node_id in child_node_ids:
    w.extend( struct.pack( '<I', child_node_id ) )
  return bytes( w )


def vox_nshp_content( node_id: int, model_id: int ) -> bytes:
  """Produces the content for the nSHP chunk, which is a shape node in the scene
  graph. A shape node represents an instance of a model.
  """
  w = bytearray( )
  w.extend( struct.pack( '<I', node_id ) )
  w.extend( vox_dict( [] ) ) # node attributes
  w.extend( struct.pack( '<I', 1 ) ) # num models
  w.extend( struct.pack( '<I', model_id ) )
  w.extend( vox_dict( [] ) ) # model attributes
  return bytes( w )


def vox_dict( entries: List[ Tuple[ str, str ] ] ) -> bytes:
  """Produces the binary representation of a dictionary for the .vox format.
  Note that all keys and values are strings.

  Examples
  --------
  >>> vox_dict( [ ( '_t', '10 5 2' ) ] )

  This dictionary (from the 'nTRN' chunk) defines a translation.
  """
  w = bytearray( )
  w.extend( struct.pack( '<I', len( entries ) ) )
  for (key, value) in entries:
    key_b = bytes( key, 'UTF-8' )
    value_b = bytes( value, 'UTF-8' )

    w.extend( struct.pack( '<I', len( key_b ) ) )
    w.extend( key_b )
    w.extend( struct.pack( '<I', len( value_b ) ) )
    w.extend( value_b )
  return bytes( w )


################################################################################
# # # # # # # # # # # # # # # # # # # MAIN # # # # # # # # # # # # # # # # # # #
################################################################################


if __name__ == '__main__':
  ## CONFIGURATION ##

  # The city's floor plan dimensions
  # MagicaVoxel supports 15 adjacent 126-wide blocks + a 110-wide block.
  WIDTH  = 15 * 126 + 110 # 2000
  DEPTH  = 15 * 126 + 110 # 2000
  # A "big street" is a street between city blocks
  BIG_STREET_SIZE     = 8
  SMALL_STREET_SIZE   = 2
  # City blocks are properly aligned. Each city blocks contains many buildings
  MIN_BIG_BLOCK_SIZE  = 30
  MAX_BIG_BLOCK_SIZE  = 100
  # The bounds on the building floor plan size along either axis
  MIN_BUILDING_SIZE   = 5
  MAX_BUILDING_SIZE   = 13
  # The bounds on the building heights. Note that buildings always have an odd
  # height. Every other row can have windows. The very top and bottom rows
  # cannot.
  MIN_BUILDING_HEIGHT = 5
  MAX_BUILDING_HEIGHT = 29

  OUT_PNG_FILE = 'city.png' # File for the floor plan
  OUT_VOX_FILE = 'city.vox' # File for the voxel mesh

  # Use this to reproduce results. This seed is used for the banner image
  SEED = 0x12345678


  ## CONTENT ##
  
  print( "This script generates a {}x{}x{} voxel city.".format( WIDTH, DEPTH, MAX_BUILDING_HEIGHT + 1 ) )
  print( "It writes a top-view image to '{}'.".format( OUT_PNG_FILE ) )
  print( "It writes the voxel mesh to '{}'. (~148MiB)".format( OUT_VOX_FILE ) )
  print( "The used seed is: 0x{:08x}".format( SEED ) )
  print( "Note that this process takes a while. On my machine it took about 70 seconds." )

  seed( SEED )

  start_time = start_phase_time = time.time( )

  # Generate the rows and columns for the city blocks
  width_bins = generate_bins( WIDTH, MIN_BIG_BLOCK_SIZE, MAX_BIG_BLOCK_SIZE, BIG_STREET_SIZE )
  depth_bins = generate_bins( DEPTH, MIN_BIG_BLOCK_SIZE, MAX_BIG_BLOCK_SIZE, BIG_STREET_SIZE )

  if width_bins is None or depth_bins is None:
    raise Exception( "City blocks could not be generated with the current settings" )

  img = Image.new( 'RGB', ( WIDTH, DEPTH ), 'white' )
  draw = ImageDraw.Draw( img )

  voxels = VoxelBuilder( WIDTH, DEPTH, MAX_BUILDING_HEIGHT + 1 )
  voxels.add_to_palette( WINDOW_COLOR1 )
  voxels.add_to_palette( WINDOW_COLOR2 )
  voxels.add_to_palette( WINDOW_COLOR3 )

  # The gray floor for the roads
  voxels.set( 0, 0, 0, WIDTH, DEPTH, 1, (128, 128, 128, 255) )
  
  # Loop over the city blocks. Within each city block, generate the buildings
  # individually
  x_loc = 0
  for width_bin in width_bins:
    y_loc = 0
    for depth_bin in depth_bins:
      # This is the city block with bounds (x_loc, y_loc, width_bin, depth_bin)

      # Generate the floor plan for the buildings
      building_floors = generate_buildings( width_bin, depth_bin, MIN_BUILDING_SIZE, MAX_BUILDING_SIZE, SMALL_STREET_SIZE )

      if building_floors is None:
        raise Exception( "Buildings could not be generated with the current settings" )

      # Loop over each building, draw it to the floorplan and add the building
      # to the voxel mesh.
      for (bx,by,bw,bh) in building_floors:
        color = rand_building_color( )
        height = randodd( MIN_BUILDING_HEIGHT, MAX_BUILDING_HEIGHT )

        # Render the building (on both the floorplan and voxel mesh)
        voxels.set( x_loc + bx, y_loc + by, 1, bw, bh, height, color )
        draw.rectangle( (x_loc+bx, y_loc+by, x_loc+bx+bw-1, y_loc+by+bh-1 ), fill = color )
        
        # Add windows. A window can have either of three colors, or be entirely
        # absent. Not every building side has windows; determine this randomly.
        if randint(0,1) == 0: # North
          for z in range( 2, height, 2 ):
            for ix in range( x_loc + bx + 1, x_loc + bx + bw - 1):
              c = rand_window_color( )
              if c is not None: # No window
                voxels.set( ix, y_loc + by, z, 1, 1, 1, c )
        if randint(0,1) == 0: # South
          for z in range( 2, height, 2 ):
            for ix in range( x_loc + bx + 1, x_loc + bx + bw - 1):
              c = rand_window_color( )
              if c is not None: # No window
                voxels.set( ix, y_loc + by + bh - 1, z, 1, 1, 1, c )
        if randint(0,1) == 0: # East
          for z in range( 2, height, 2 ):
            for iy in range( y_loc + by + 1, y_loc + by + bh - 1):
              c = rand_window_color( )
              if c is not None: # No window
                voxels.set( x_loc + bx, iy, z, 1, 1, 1, c )
        if randint(0,1) == 0: # West
          for z in range( 2, height, 2 ):
            for iy in range( y_loc + by + 1, y_loc + by + bh - 1):
              c = rand_window_color( )
              if c is not None: # No window
                voxels.set( x_loc + bx + bw - 1, iy, z, 1, 1, 1, c )

      y_loc = y_loc + depth_bin + BIG_STREET_SIZE

    x_loc = x_loc + width_bin + BIG_STREET_SIZE
  # end for-loop over city blocks

  print( "Done generating the voxel mesh ({} seconds)".format( round( time.time( ) - start_phase_time ) ) )

  print( "Writing PNG to file '{}'".format( OUT_PNG_FILE ) )
  img.show( )
  img.save( OUT_PNG_FILE )
  print( "PNG successfully written" )

  print( "Constructing binary .vox representation" )
  start_phase_time = time.time( )
  bs = voxels.to_vox( )

  print( "Writing VOX to file '{}' ({} seconds)".format( OUT_VOX_FILE, round( time.time() - start_phase_time ) ) )
  f = open( OUT_VOX_FILE, 'wb' )
  f.write( bs )
  f.close( )

  elapsed_time = time.time() - start_time
  print( "Done. Total elapsed time: {} seconds".format( round( elapsed_time ) ) )
