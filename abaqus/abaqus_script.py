from part import *
from material import *
from section import *
from assembly import *
from step import *
from interaction import *
from load import *
from mesh import *
from optimization import *
from job import *
from sketch import *
from visualization import *
from connectorBehavior import *
from odbAccess import *

import os
import argparse
import numpy as np

parser = argparse.ArgumentParser(description="Evaluate the stress-strain response of given pixel-based structures.")
parser.add_argument('--samples_path', type=str, required=False, default='./', help='Path to samples to evaluate.')
parser.add_argument('--sample_index', type=int, required=True, help='Index of sample to evaluate.')
parser.add_argument('--store_frames', type=bool, required=False, default=False, help='Store frames of simulation as np array. Takes some time.')
parser.add_argument('--pixels', type=int, required=False, default=48, help='Number of pixels')
parser.add_argument('--strain', type=float, required=False, default=-0.2, help='Strain value')
parser.add_argument('--numEvalIntervals', type=int, required=False, default=50, help='Number of evaluation intervals')
parser.add_argument('--bounding_box', type=bool, required=False, default=False, help='Consider bounding box')
parser.add_argument('--radius', type=float, required=False, default=0., help='Radius of splined geometry, increase value for smoother boundaries.')
parser.add_argument('--density', type=float, required=False, default=1.e-8)
parser.add_argument('--coarseness', type=float, required=False, default=0.015)
args, unknown = parser.parse_known_args()

samples_path = args.samples_path
sample_index = args.sample_index
store_frames = args.store_frames
pixels = args.pixels
bounding_box = args.bounding_box
radius = args.radius
density = args.density
strain = args.strain
coarseness = args.coarseness
numEvalIntervals = args.numEvalIntervals

dist_boundary = 0.00
tesselations = 1

strain_spacing = np.linspace(0., np.abs(strain), num = numEvalIntervals+1)

# y_periodicity = True
y_periodicity = False

make_periodic = True
splined = True

material = 'Jin_et_al' # https://doi.org/10.1073/pnas.1913228117

# solver = 'static'
solver = 'implicit'
# solver = 'explicit'

# size of window from which pixels are stored
domain_length = 1.0
# notation
model_name = 'Model-1'
part_name = 'Part-1'
material_name = 'Material-1'
job_name = 'Job-1'

if samples_path == './':
    save_path = './csv/'
else:
    save_path = os.path.join(samples_path, 'abaqus', 'csv')
    try:
        os.makedirs(save_path)
    except OSError as e:
        if e.errno != os.errno.EEXIST:  # Check if the error is "File exists"
            raise  # Re-raise the exception if it's not "File exists"

m = mdb.models[model_name]

# convert pixel topology to splined Abaqus part, requires some hacks
def create_part(geom, pixels, radius=0):
    elements_per_pixel = 2 # must at least be 2
    elements = pixels*elements_per_pixel

    nx, ny = (elements+1, elements+1)
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    xv, yv = np.meshgrid(x, y)
    coords = np.stack((xv.flatten(),yv.flatten()),axis=1)

    full_edges = np.zeros((2*elements*(elements+1),2), dtype=int)

    temp = 0
    for i in range(elements+1):
        for j in range(elements):
            full_edges[temp,:] = [i*(elements+1)+j,i*(elements+1)+j+1]
            temp += 1
    for i in range(elements+1):
        for j in range(elements):
            full_edges[temp,:] = [j*(elements+1)+i,(j+1)*(elements+1)+i]
            temp += 1

    # if elements_per_pixel > 1:
    geom = np.repeat(np.repeat(geom.reshape(pixels,pixels),elements_per_pixel,axis=1),elements_per_pixel,axis=0)

    diff_x = np.diff(geom,axis=0)
    diff_y = np.diff(geom,axis=1)

    full_diff_x_rev = np.concatenate((geom[0,:][None],diff_x,geom[-1,:][None]),axis=0)
    full_diff_x = full_diff_x_rev[::-1,:]
    full_diff_y_rev = np.concatenate((geom[:,0][:,None],diff_y,geom[:,-1][:,None]),axis=1)
    full_diff_y = full_diff_y_rev[::-1,:]

    true_edges_bin = np.abs(np.concatenate((full_diff_x.flatten(),np.transpose(full_diff_y).flatten()),axis=0))
    true_edges_bin_idx = np.argwhere(true_edges_bin==True).flatten()

    true_edges = full_edges[true_edges_bin_idx]

    full_permutations = []

    while true_edges.shape[0] != 0:
        flag1 = True
        flag2 = True
        cur_permutation = []
        while flag2:
            if flag1 == True:
                cur_permutation.append(true_edges[0,0])
                cur_permutation.append(true_edges[0,1])
                cur_val = true_edges[0,1]
                true_edges = np.delete(true_edges, 0, axis=0)
                flag1 = False
            idx1 = np.where(true_edges == cur_val)[0][0]
            idx2 = np.where(true_edges == cur_val)[1][0]
            cur_permutation.append(true_edges[idx1,idx2])
            cur_permutation.append(true_edges[idx1,1-idx2])
            cur_val = true_edges[idx1,1-idx2]
            true_edges = np.delete(true_edges, idx1, axis=0)
            if len(np.where(true_edges == cur_val)[0]) == 0:
                flag2 = False
        full_permutations.append(cur_permutation)
        flag2 = True
        flag1 = True

    red_permutations = []

    for list in full_permutations:
        red_list = []
        for iter, i in enumerate(list):
            if iter == 0 or iter == len(list)-1:
                cur_pos = coords[i]
                temp = i
                red_list.append(temp)
                continue
            else:
                if ((not np.isclose(coords[i,0], cur_pos[0])) 
                and (not np.isclose(coords[i,1], cur_pos[1]))):
                    cur_pos = coords[i]
                    red_list.append(temp)
                temp = i
        red_permutations.append(red_list)

    m.ConstrainedSketch(name='__profile__', sheetSize=2.0)

    full_surf_list = []
    top_surf_list = []
    bot_surf_list = []
    outer_surf_no_boundary_list = []
    inner_node_list = []

    longest_permutation = 0

    for idx, list in enumerate(red_permutations):
        permutations_np = np.array(list)

        cur_surf_list = []

        spline_list = []

        longest_permutation_flag = False
        if len(permutations_np)-1 > longest_permutation:
            longest_permutation = len(permutations_np)-1
            longest_permutation_flag = True
            outer_surf_no_boundary_list = []

        for i in range(len(permutations_np)-1):

            point1x = coords[permutations_np[i],0]
            point1y = coords[permutations_np[i],1]

            point2x = coords[permutations_np[i+1],0]
            point2y = coords[permutations_np[i+1],1]

            point1 = np.array([point1x,point1y])
            point2 = np.array([point2x,point2y])
            v_12 = point2 - point1

            midpoint = point1 + 0.5*v_12

            cur_surf_list.append(((midpoint[0], midpoint[1], 0.),))

            if idx == 0 and i==0:
                inner_node_list.append(((midpoint[0]+1.e-3, midpoint[1]+1.e-3, 0.),))

            if np.isclose(midpoint[1], 1.):
                top_surf_list.append(((midpoint[0], midpoint[1], 0.),))
            elif np.isclose(midpoint[1], 0.):
                bot_surf_list.append(((midpoint[0], midpoint[1], 0.),))
            elif longest_permutation_flag:
                outer_surf_no_boundary_list.append(((midpoint[0], midpoint[1], 0.),))

            if splined:
                # case distinguishment
                point1_boundary = False
                point2_boundary = False

                if idx == 0:
                    if ((np.isclose(point1x, 0.) or np.isclose(point1x, 1.)) or
                        (np.isclose(point1y, 0.) or np.isclose(point1y, 1.))):
                        point1_boundary = True

                    if ((np.isclose(point2x, 0.) or np.isclose(point2x, 1.)) or
                        (np.isclose(point2y, 0.) or np.isclose(point2y, 1.))):
                        point2_boundary = True

                    if point1_boundary and point2_boundary:
                        m.sketches['__profile__'].Line(point1=(point1[0], point1[1]), 
                            point2=(point2[0], point2[1]))
                    elif point1_boundary and not point2_boundary:
                        spline_list.append(((point1[0],point1[1],)))
                    elif not point1_boundary and point2_boundary:
                        spline_list.append(((point2[0],point2[1],)))
                        m.sketches['__profile__'].Spline(points=(spline_list))
                        spline_list = []
                    elif not point1_boundary and not point2_boundary:
                        spline_list.append(((midpoint[0],midpoint[1],)))
                else:
                    if i == 0:
                        init_midpoint = midpoint
                    spline_list.append(((midpoint[0],midpoint[1],)))
                    if i == len(permutations_np)-2:
                        spline_list.append(((init_midpoint[0],init_midpoint[1],)))
                        m.sketches['__profile__'].Spline(points=(spline_list))

            else:
                if radius > 0:
                    if i == len(permutations_np)-2:
                        point3x = coords[permutations_np[1],0]
                        point3y = coords[permutations_np[1],1]
                    else:
                        point3x = coords[permutations_np[i+2],0]
                        point3y = coords[permutations_np[i+2],1]
                    point3 = np.array([point3x,point3y])
                    v_23 = point3 - point2

                    if radius-1.e-5 > np.linalg.norm(v_12)/2 or radius-1.e-5 > np.linalg.norm(v_23)/2:
                        raise ValueError('Radius too large for given mesh.')

                    v_12_hat = v_12/np.linalg.norm(v_12)
                    v_23_hat = v_23/np.linalg.norm(v_23)

                    point1r = point1 + radius*v_12_hat
                    point2r = point2 - radius*v_12_hat
                    point3r = point2 + radius*v_23_hat

                    center = point2 - radius*v_12_hat + radius*v_23_hat

                    direction_vec = np.cross(-v_12, v_23)
                    if direction_vec > 0:
                        direction = CLOCKWISE
                    else:
                        direction = COUNTERCLOCKWISE

                    m.sketches['__profile__'].Line(point1=(point1r[0], point1r[1]), 
                        point2=(point2r[0], point2r[1]))

                    m.sketches['__profile__'].ArcByCenterEnds(center=(center[0],center[1]), 
                        direction=direction, point1=(point2r[0], point2r[1]), point2=(point3r[0], point3r[1]))

                else:
                    m.sketches['__profile__'].Line(point1=(point1[0], point1[1]), 
                        point2=(point2[0], point2[1]))
        
        full_surf_list.append(cur_surf_list)

    m.Part(dimensionality=TWO_D_PLANAR, name=part_name, type=
        DEFORMABLE_BODY)
    m.parts[part_name].BaseShell(sketch=
        m.sketches['__profile__'])
    del m.sketches['__profile__']

    return full_surf_list, top_surf_list, bot_surf_list, outer_surf_no_boundary_list, inner_node_list

# load geometry
geometries = np.genfromtxt(os.path.join(samples_path, 'geometries.csv'), delimiter=',').reshape(-1,pixels,pixels)
selected_geometry = geometries[sample_index]

if bounding_box == 'True':
    for i in range(selected_geometry.shape[0]):
        for j in range(selected_geometry.shape[1]):
            if i == 0 or i == selected_geometry.shape[0]-1 or j == 0 or j == selected_geometry.shape[0]-1:
                selected_geometry[i,j] = 1

# 2-fold mirroring of geometry to make periodic
if make_periodic:
    geom_dr = np.flip(selected_geometry,0)
    geom_ul = np.flip(selected_geometry,1)
    geom_ur = np.flip(geom_ul,0)
    geom_d = np.concatenate((selected_geometry,geom_dr), axis=0)
    geom_u = np.concatenate((geom_ul,geom_ur), axis=0)
    geom_periodic = np.concatenate((geom_u, geom_d), axis=1)
    cad_pixels = 2 * pixels
else:
    cad_pixels = pixels
    geom_periodic = selected_geometry

# consider multiple tesselations
geom = np.tile(np.transpose(np.tile(geom_periodic,tesselations)),tesselations)

# create part
full_surf_list, top_surf_list, bot_surf_list, outer_surf_no_boundary_list, inner_node_list = create_part(geom, cad_pixels*tesselations, radius)

# identify full surface and full set
for idx, list in enumerate(full_surf_list):
    m.parts[part_name].Surface(name='Surf_'+str(idx), side1Edges=
        m.parts[part_name].edges.findAt(*tuple(list)))
num_surf = len(full_surf_list)

m.parts[part_name].Surface(name='Surf_outer', side1Edges=
    m.parts[part_name].edges.findAt(*tuple(outer_surf_no_boundary_list)))

# create constitutive model
if material == 'Jin_et_al':
    m.Material(name=material_name)
    E = 2.306e3
    nu = 0.35
    m.materials[material_name].Elastic(table=((E, nu), ))

    m.materials[material_name].Density(table=((density, ), ))
    m.materials[material_name].Damping(beta=0.005)

    m.materials[material_name].Plastic(table=( (40.62, 0.0),
                        (45.24, 0.001133),
                        (52.62, 0.004183),
                        (58.00, 0.0080645),
                        (61.87, 0.012557),
                        (65.81, 0.020035),
                        (69.19, 0.030689),
                        (71.06, 0.038873),
                        (72.61, 0.047114),
                        (73.54, 0.052610),
                        (74.82, 0.06083),
                        (76.74, 0.074477),
                        (78.46, 0.08799),
                        (81.58, 0.11457),
                        (83.00, 0.1276)
                        ))
else:
    m.Material(name=material_name)
    m.materials[material_name].Density(table=((density,), ))
    m.materials[material_name].Hyperelastic(materialType=
        ISOTROPIC, table=((0.192311, 0.288461), ), testData=OFF, type=NEO_HOOKE, 
        volumetricResponse=VOLUMETRIC_DATA)

m.HomogeneousSolidSection(material=material_name, name=
    'Section-1', thickness=None)
    
m.parts[part_name].Set(faces=
    m.parts[part_name].faces.findAt((tuple(inner_node_list)[0])), name='FullSet')

m.parts[part_name].SectionAssignment(offset=0.0, 
    offsetField='', offsetType=MIDDLE_SURFACE, region=
    m.parts[part_name].sets['FullSet'], sectionName=
    'Section-1', thicknessAssignment=FROM_SECTION)
    
# seed mesh
m.parts[part_name].seedPart(deviationFactor=0.1, 
    minSizeFactor=0.1, size=coarseness)
m.parts[part_name].generateMesh()

if solver == 'static':
    # set element to plain strain
    m.parts[part_name].setElementType(elemTypes=(ElemType(
        elemCode=CPE4, elemLibrary=STANDARD, secondOrderAccuracy=OFF, 
        hourglassControl=DEFAULT, distortionControl=DEFAULT), ElemType(
        elemCode=CPE3, elemLibrary=STANDARD)), regions=(
        m.parts[part_name].faces.findAt((tuple(inner_node_list)[0])), ))
else:
    # set element to plain strain
    m.parts[part_name].setElementType(elemTypes=(ElemType(
        elemCode=CPE4R, elemLibrary=STANDARD, secondOrderAccuracy=OFF, 
        hourglassControl=DEFAULT, distortionControl=DEFAULT), ElemType(
        elemCode=CPE3, elemLibrary=STANDARD)), regions=(
        m.parts[part_name].faces.findAt((tuple(inner_node_list)[0])), ))

# create top and bottom line for compression
m.ConstrainedSketch(name='__profile__', sheetSize=4.0)
m.sketches['__profile__'].Line(point1=(-1.0, 1.0 + dist_boundary), point2=(
    2.0, 1.0+dist_boundary))
m.sketches['__profile__'].geometry.findAt((0.5, 1.0+dist_boundary))
m.sketches['__profile__'].HorizontalConstraint(
    addUndoState=False, entity=
    m.sketches['__profile__'].geometry.findAt((0.5, 1.0+dist_boundary), 
    ))
m.Part(dimensionality=TWO_D_PLANAR, name='TopLine', type=
    ANALYTIC_RIGID_SURFACE)
m.parts['TopLine'].AnalyticRigidSurf2DPlanar(sketch=
    m.sketches['__profile__'])
del m.sketches['__profile__']
m.parts['TopLine'].ReferencePoint(point=
    m.parts['TopLine'].InterestingPoint(
    m.parts['TopLine'].edges.findAt((-0.25, 1.0+dist_boundary, 0.0), ), 
    MIDDLE))

m.parts['TopLine'].Surface(name='TopSurf', side2Edges=
    m.parts['TopLine'].edges.findAt(((-0.25, 1.0+dist_boundary, 0.0), )))

m.ConstrainedSketch(name='__profile__', sheetSize=4.0)
m.sketches['__profile__'].Line(point1=(-1.0, 0.0 - dist_boundary), point2=(
    2.0, 0.0-dist_boundary))
m.sketches['__profile__'].geometry.findAt((0.5, 0.0-dist_boundary))
m.sketches['__profile__'].HorizontalConstraint(
    addUndoState=False, entity=
    m.sketches['__profile__'].geometry.findAt((0.5, 0.0-dist_boundary), 
    ))
m.Part(dimensionality=TWO_D_PLANAR, name='BotLine', type=
    ANALYTIC_RIGID_SURFACE)
m.parts['BotLine'].AnalyticRigidSurf2DPlanar(sketch=
    m.sketches['__profile__'])
del m.sketches['__profile__']
m.parts['BotLine'].ReferencePoint(point=
    m.parts['BotLine'].InterestingPoint(
    m.parts['BotLine'].edges.findAt((-0.25, 0.0-dist_boundary, 0.0), ), 
    MIDDLE))
m.parts['BotLine'].Surface(name='BotSurf', side1Edges=
    m.parts['BotLine'].edges.findAt(((-0.25, 0.0-dist_boundary, 0.0), )))

# assemble part
m.rootAssembly.DatumCsysByDefault(CARTESIAN)
m.rootAssembly.Instance(dependent=ON, name='BotLine-1', 
    part=m.parts['BotLine'])
m.rootAssembly.Instance(dependent=ON, name='TopLine-1', 
    part=m.parts['TopLine'])
m.rootAssembly.Instance(dependent=ON, name='Part-1-1', 
    part=m.parts[part_name])

# create node sets for bcs
bot_nodes = []
top_nodes = []
corner_nodes = []
top_corner_nodes = []
bot_corner_nodes = []

allNodes = m.rootAssembly.instances['Part-1-1'].nodes
for node in allNodes:
    if np.isclose(node.coordinates[1], 0.):
        bot_nodes.append(node.label-1)
    if np.isclose(node.coordinates[1], 1.):
        top_nodes.append(node.label-1)
    if (np.isclose(node.coordinates[0], 0.) and np.isclose(node.coordinates[1], 0.)
     or np.isclose(node.coordinates[0], 1.) and np.isclose(node.coordinates[1], 0.)
     or np.isclose(node.coordinates[0], 0.) and np.isclose(node.coordinates[1], 1.)
     or np.isclose(node.coordinates[0], 1.) and np.isclose(node.coordinates[1], 1.)):
        corner_nodes.append(node.label-1)
    if (np.isclose(node.coordinates[0], 0.) and np.isclose(node.coordinates[1], 1.)
     or np.isclose(node.coordinates[0], 1.) and np.isclose(node.coordinates[1], 1.)):
        top_corner_nodes.append(node.label-1)
    if (np.isclose(node.coordinates[0], 0.) and np.isclose(node.coordinates[1], 0.)
     or np.isclose(node.coordinates[0], 1.) and np.isclose(node.coordinates[1], 0.)):
        bot_corner_nodes.append(node.label-1)

botNodes = [m.rootAssembly.instances['Part-1-1'].nodes[idx:idx+1] for idx in bot_nodes]
topNodes = [m.rootAssembly.instances['Part-1-1'].nodes[idx:idx+1] for idx in top_nodes]
m.rootAssembly.Set(name='BotNodesSet', nodes=botNodes)
m.rootAssembly.Set(name='TopNodesSet', nodes=topNodes)

m.rootAssembly.Set(name='TopLineSet', referencePoints=(
    m.rootAssembly.instances['TopLine-1'].referencePoints[2], 
    ))

# solver
if solver == 'explicit':
    m.ExplicitDynamicsStep(improvedDtMethod=ON, name='Step-1', 
        previous='Initial')
elif solver == 'implicit':
    m.ImplicitDynamicsStep(name='Step-1',
            previous='Initial',
            timePeriod=1.0,
            nlgeom=ON,
            initialInc=1e-3,
            minInc=1e-8,
            maxNumInc=400,
            alpha=DEFAULT,
            amplitude=RAMP,
            application=MODERATE_DISSIPATION,
            initialConditions=OFF)
elif solver == 'static':
    m.StaticStep(name='Step-1', 
        nlgeom=ON,
        previous='Initial')
    ALE_adaptive_mesh = True
    if ALE_adaptive_mesh:
        m.AdaptiveMeshControl(name='Ada-1')
        m.steps['Step-1'].AdaptiveMeshDomain(controls='Ada-1', 
        region=m.rootAssembly.instances['Part-1-1'].sets['FullSet'],
        frequency=5,
        meshSweeps=3)

# establish contact model
m.ContactProperty('IntProp-1')
m.interactionProperties['IntProp-1'].TangentialBehavior(
    dependencies=0, directionality=ISOTROPIC, elasticSlipStiffness=None, 
    formulation=PENALTY, fraction=0.005, maximumElasticSlip=FRACTION, 
    pressureDependency=OFF, shearStressLimit=None, slipRateDependency=OFF, 
    table=((0.4, ), ), temperatureDependency=OFF)
m.interactionProperties['IntProp-1'].NormalBehavior(
    allowSeparation=ON, constraintEnforcementMethod=DEFAULT, 
    pressureOverclosure=HARD)
m.ContactProperty('SlipContact')
m.interactionProperties['SlipContact'].NormalBehavior(
    allowSeparation=ON, constraintEnforcementMethod=DEFAULT, 
    pressureOverclosure=HARD)

if solver == 'explicit':
    # self-contact
    for idx in range(num_surf):
        m.SelfContactExp(createStepName='Step-1', 
            interactionProperty='IntProp-1', mechanicalConstraint=KINEMATIC, name=
            'Int-1-' + str(idx+1), surface=
            m.rootAssembly.instances['Part-1-1'].surfaces['Surf_' + str(idx)])
    if not y_periodicity:
        # top contact
        m.SurfaceToSurfaceContactExp(
            clearanceRegion=None, createStepName='Step-1', datumAxis=None, 
            initialClearance=OMIT, interactionProperty='SlipContact', master=
            m.rootAssembly.instances['TopLine-1'].surfaces['TopSurf']
            , name='TopContact', slave=
            m.rootAssembly.instances['Part-1-1'].surfaces['Surf_0']
            , sliding=FINITE)
        # bot contact
        m.SurfaceToSurfaceContactExp(
            clearanceRegion=None, createStepName='Step-1', datumAxis=None, 
            initialClearance=OMIT, interactionProperty='SlipContact', master=
            m.rootAssembly.instances['BotLine-1'].surfaces['BotSurf']
            , name='BotContact', slave=
            m.rootAssembly.instances['Part-1-1'].surfaces['Surf_0']
            , sliding=FINITE)
elif solver == 'implicit' or solver == 'static':
    # self-contact
    for idx in range(num_surf):
        m.SelfContactStd(createStepName='Step-1', 
        interactionProperty='IntProp-1', name='Int-1-' + str(idx+1), surface=
        m.rootAssembly.instances['Part-1-1'].surfaces['Surf_' + str(idx)], thickness=ON)
    if not y_periodicity:
        # top contact
        m.SurfaceToSurfaceContactStd(adjustMethod=NONE, 
            clearanceRegion=None, createStepName='Step-1', datumAxis=None, 
            initialClearance=OMIT, interactionProperty='SlipContact', master=
            m.rootAssembly.instances['TopLine-1'].surfaces['TopSurf']
            , name='TopContact', slave=
            m.rootAssembly.instances['Part-1-1'].surfaces['Surf_0']
            , sliding=FINITE, thickness=ON)
        # bot contact
        m.SurfaceToSurfaceContactStd(adjustMethod=NONE, 
            clearanceRegion=None, createStepName='Step-1', datumAxis=None, 
            initialClearance=OMIT, interactionProperty='SlipContact', master=
            m.rootAssembly.instances['BotLine-1'].surfaces['BotSurf']
            , name='BotContact', slave=
            m.rootAssembly.instances['Part-1-1'].surfaces['Surf_0']
            , sliding=FINITE, thickness=ON)

# establish field output recordings
        
# we sample one additional step to mitigate inertia effects
strain += strain/numEvalIntervals
numEvalIntervals += 1

# sample every 0.1 strain steps (note that we apply smooth amplitude in time to further surpress inertia, hence strain is not equidistant)
# consider 11 points below, note that we consider a strain of 0.01 at the beginning and scale everything with (50/51) to be consistent
m.TimePoint(name='TimePoints-1', points=(
 (0.1049, ), (0.24473, ), (0.32388, ), (0.38634, ), (0.44196, ), (0.49477, ), (0.54734, ),
 (0.60215, ), (0.66277, ), (0.73705, ), (0.86569, )
 ))

# consider 52 points below since we include 0 and one additional strain step at the end which we remove later
m.TimePoint(name='TimePoints-2', points=(
 (0.0, ), (0.13431, ), (0.17291, ), (0.20117, ), (0.22446, ), (0.24473, ), (0.26295, ), (0.27968, ),
 (0.29526, ), (0.30994, ), (0.32388, ), (0.33723, ), (0.35008, ), (0.36251, ), (0.37458, ), (0.38634, ),
 (0.39785, ), (0.40913, ), (0.42023, ), (0.43116, ), (0.44196, ), (0.45266, ), (0.46327, ), (0.47381, ),
 (0.4843, ), (0.49477, ), (0.50523, ), (0.5157, ), (0.52619, ), (0.53673, ), (0.54734, ), (0.55804, ),
 (0.56884, ), (0.57977, ), (0.59087, ), (0.60215, ), (0.61366, ), (0.62542, ), (0.63749, ), (0.64992, ),
 (0.66277, ), (0.67612, ), (0.69006, ), (0.70474, ), (0.72032, ), (0.73705, ), (0.75527, ), (0.77554, ),
 (0.79883, ), (0.82709, ), (0.86569, ), (1.0, ))
 )

m.fieldOutputRequests['F-Output-1'].setValues(timePoint=
    'TimePoints-1', variables=('S', 'PE', 'PEEQ', 'PEMAG', 'LE', 'U', 
    'RF', 'CSTRESS', 'CDISP', 'ENER', 'ELEN', 'ELEDEN', 'DAMAGEC', 'SENER',
    'DAMAGET', 'SDEG', 'COORD'))

# establish displacement and reaction force recording
del m.historyOutputRequests['H-Output-1']
m.HistoryOutputRequest(createStepName='Step-1', name=
    'H-Output-0', rebar=EXCLUDE, region=
    m.rootAssembly.sets['TopNodesSet'], sectionPoints=DEFAULT, 
    variables=('U2', 'RF2'), timePoint= 'TimePoints-2')
m.HistoryOutputRequest(createStepName='Step-1', name=
    'H-Output-1', rebar=EXCLUDE, region=
    m.rootAssembly.sets['TopLineSet'], sectionPoints=DEFAULT, 
    variables=('U2', 'RF2'), timePoint= 'TimePoints-2')

# establish smooth amplitude
m.SmoothStepAmplitude(data=((0.0, 0.0), (1.0, 1.0)), name= 'Amp-1', timeSpan=STEP)

if solver == 'explicit':
    m.HistoryOutputRequest(createStepName='Step-1', name=
        'H-Output-2', timePoint= 'TimePoints-2',
        variables=('ALLAE', 'ALLKE', 'ALLIE', 'ETOTAL'))
else:
    m.HistoryOutputRequest(createStepName='Step-1', name=
        'H-Output-2', timePoint= 'TimePoints-2', 
        variables=('ALLAE', 'ALLSD', 'ALLKE', 'ALLIE', 'ETOTAL'))

if corner_nodes:
    cornerNodes = [m.rootAssembly.instances['Part-1-1'].nodes[idx:idx+1] for idx in corner_nodes]
    m.rootAssembly.Set(name='CornerNodesSet', nodes=cornerNodes)
    cornerNodes = [m.rootAssembly.instances['Part-1-1'].nodes[idx:idx+1] for idx in bot_corner_nodes]
    m.rootAssembly.Set(name='BotCornerNodesSet', nodes=cornerNodes)
    cornerNodes = [m.rootAssembly.instances['Part-1-1'].nodes[idx:idx+1] for idx in top_corner_nodes]
    m.rootAssembly.Set(name='TopCornerNodesSet', nodes=cornerNodes)

flag_RBM_x = False
flag_RBM_y = False

if corner_nodes and y_periodicity:
    m.DisplacementBC(amplitude=UNSET, createStepName='Initial', 
        distributionType=UNIFORM, fieldName='', localCsys=None, name='RBM_X_TopCorner', 
        region=m.rootAssembly.sets['TopCornerNodesSet'], u1=SET, u2=UNSET,
        ur3=UNSET)
    # if periodic in y-direction, we fix also the bottom corners in y and apply a the strain for top corners
    # we can do this since we decide to fix the DOF of the corner node (we can pick any node to apply to DOF to fix RBM)
    m.DisplacementBC(amplitude=UNSET, createStepName='Initial', 
        distributionType=UNIFORM, fieldName='', localCsys=None, name='RBM_XY_BotCorner', 
        region=m.rootAssembly.sets['BotCornerNodesSet'], u1=SET, u2=SET,
        ur3=UNSET)
    m.DisplacementBC(amplitude='Amp-1', createStepName='Step-1', 
        distributionType=UNIFORM, fieldName='', fixed=OFF, localCsys=None, name='RBM_Y_TopCorner', 
        region=m.rootAssembly.sets['TopCornerNodesSet'], u1=UNSET, u2=strain,
        ur3=UNSET)
    flag_RBM_x = True
    flag_RBM_y = True

periodic_bc_flag = False

# set bcs
m.rootAssembly.Set(name='BotLineSet', referencePoints=(
    m.rootAssembly.instances['BotLine-1'].referencePoints[2], 
    ))
# fix bottom line in x and y and rotation
m.DisplacementBC(amplitude=UNSET, createStepName='Initial',
    distributionType=UNIFORM, fieldName='', localCsys=None, name='BC-1', 
    region=m.rootAssembly.sets['BotLineSet'], u1=SET, u2=SET, 
    ur3=SET)
# fix top line in x and rotation
m.DisplacementBC(amplitude=UNSET, createStepName='Initial', 
    distributionType=UNIFORM, fieldName='', localCsys=None, name='BC-2', 
    region=m.rootAssembly.sets['TopLineSet'], u1=SET, u2=UNSET
    , ur3=SET)
# apply strain in y for top line
m.rootAssembly.Set(name='TopLineSet', referencePoints=(
    m.rootAssembly.instances['TopLine-1'].referencePoints[2], 
    ))
m.DisplacementBC(amplitude='Amp-1', createStepName='Step-1'
    , distributionType=UNIFORM, fieldName='', fixed=OFF, localCsys=None, name=
    'BC-3', region=m.rootAssembly.sets['TopLineSet'], u1=UNSET
    , u2=strain, ur3=UNSET)

if not y_periodicity:
    # we 'attach' top and bottom nodes in x and y to plate
    m.DisplacementBC(amplitude=UNSET, createStepName='Initial',
        distributionType=UNIFORM, fieldName='', localCsys=None, name='BC-4', 
        region=m.rootAssembly.sets['BotNodesSet'], u1=SET, u2=SET, 
        ur3=UNSET)
    m.DisplacementBC(amplitude=UNSET, createStepName='Initial',
        distributionType=UNIFORM, fieldName='', localCsys=None, name='BC-5', 
        region=m.rootAssembly.sets['TopNodesSet'], u1=SET, u2=UNSET, 
        ur3=UNSET)
    m.DisplacementBC(amplitude='Amp-1', createStepName='Step-1'
        , distributionType=UNIFORM, fieldName='', fixed=OFF, localCsys=None, name=
        'BC-6', region=m.rootAssembly.sets['TopNodesSet'], u1=UNSET, 
        u2=strain, ur3=UNSET)
    flag_RBM_x = True
    flag_RBM_y = True

# apply periodic boundary conditions
delta = 1.e-3
for idx, node in enumerate(allNodes):
    if (np.isclose(node.coordinates[0], 0.) and (not np.isclose(node.coordinates[1], 0.)) and (not np.isclose(node.coordinates[1], 1.))):
        cur_node_lr = allNodes.getByBoundingBox(-delta, node.coordinates[1]-delta, 0.-delta, +delta, node.coordinates[1]+delta, 0. + delta)
        cur_node_lr_partner = allNodes.getByBoundingBox(1. - delta, node.coordinates[1]-delta, 0.-delta, 1. + delta, node.coordinates[1]+delta, 0. + delta)
        if not cur_node_lr_partner:
            periodic_bc_flag = True
        m.rootAssembly.Set(name='BC_lr_' + str(idx) + 'A', nodes=cur_node_lr)
        m.rootAssembly.Set(name='BC_lr_' + str(idx) + 'B', nodes=cur_node_lr_partner)
        if not flag_RBM_x:
            # if no corner nodes, fix first node pair in x
            m.DisplacementBC(amplitude=UNSET, createStepName='Initial', 
                distributionType=UNIFORM, fieldName='', localCsys=None, name='RBM_X_A', 
                region=m.rootAssembly.sets['BC_lr_' + str(idx) + 'A'], u1=SET, u2=UNSET
                ,ur3=UNSET)
            m.DisplacementBC(amplitude=UNSET, createStepName='Initial',
                distributionType=UNIFORM, fieldName='', localCsys=None, name='RBM_X_B',
                region=m.rootAssembly.sets['BC_lr_' + str(idx) + 'B'], u1=SET, u2=UNSET
                ,ur3=UNSET)
            m.Equation(name='Constraint-' + str(idx) + '_lr_y', terms=((1.0, 'BC_lr_' + str(idx) + 'A', 
                2), (-1.0, 'BC_lr_' + str(idx) + 'B', 2)))
            flag_RBM_x = True
        else:
            # apply periodic boundary conditions in x direction (but both for x and y displacements)
            m.Equation(name='Constraint-' + str(idx) + '_lr_x', terms=((1.0, 'BC_lr_' + str(idx) + 'A',
                1), (-1.0, 'BC_lr_' + str(idx) + 'B', 1)))
            m.Equation(name='Constraint-' + str(idx) + '_lr_y', terms=((1.0, 'BC_lr_' + str(idx) + 'A', 
                2), (-1.0, 'BC_lr_' + str(idx) + 'B', 2)))
    elif (np.isclose(node.coordinates[1], 0.) and (not np.isclose(node.coordinates[0], 0.)) and (not np.isclose(node.coordinates[0], 1.))):
        cur_node_ud = allNodes.getByBoundingBox(node.coordinates[0]-delta, 0. - delta, 0. - delta, node.coordinates[0] + delta, 0. + delta, 0. + delta)
        cur_node_ud_partner = allNodes.getByBoundingBox(node.coordinates[0] - delta, 1. - delta, 0.-delta, node.coordinates[0] + delta, 1. + delta, 0. + delta)
        if y_periodicity:
            if not cur_node_ud_partner:
                periodic_bc_flag = True
            m.rootAssembly.Set(name='BC_ud_' + str(idx) + 'A', nodes=cur_node_ud)
            m.rootAssembly.Set(name='BC_ud_' + str(idx) + 'B', nodes=cur_node_ud_partner)
            if not flag_RBM_y:
                # if no corner nodes or no periodic boundary conditions in y, fix first node pair in y
                m.DisplacementBC(amplitude=UNSET, createStepName='Initial', 
                    distributionType=UNIFORM, fieldName='', localCsys=None, name='RBM_Y_A', 
                    region=m.rootAssembly.sets['BC_ud_' + str(idx) + 'A'], u1=UNSET, u2=SET
                    ,ur3=UNSET)
                m.DisplacementBC(amplitude='Amp-1', createStepName='Step-1',
                    distributionType=UNIFORM, fieldName='', fixed=OFF, localCsys=None, name='RBM_Y_B',
                    region=m.rootAssembly.sets['BC_ud_' + str(idx) + 'B'], u1=UNSET, u2=strain
                    ,ur3=UNSET)
                m.Equation(name='Constraint-' + str(idx) + '_ud_x', terms=((1.0, 'BC_ud_' + str(idx) + 'A',
                    1), (-1.0, 'BC_ud_' + str(idx) + 'B', 1)))
                flag_RBM_y = True
            else:
                m.Equation(name='Constraint-' + str(idx) + '_ud_x', terms=((1.0, 'BC_ud_' + str(idx) + 'A',
                    1), (-1.0, 'BC_ud_' + str(idx) + 'B', 1)))
                # # NOTE we do not consider this coupling as we apply a vertical displacement
                m.Equation(name='Constraint-' + str(idx) + '_ud_y', terms=(
                    (1.0, 'BC_ud_' + str(idx) + 'A', 2),
                    (-1.0, 'BC_ud_' + str(idx) + 'B', 2),
                    (1.0, 'TopLineSet', 2)))
        else:
            pass

# create job
mdb.Job(activateLoadBalancing=False, atTime=None, contactPrint=OFF,
    description='', echoPrint=OFF, explicitPrecision=SINGLE, historyPrint=OFF,
    memory=90, memoryUnits=PERCENTAGE, model=model_name, modelPrint=OFF,
    multiprocessingMode=DEFAULT, name=job_name, nodalOutputPrecision=SINGLE,
    numCpus=1, numDomains=1, parallelizationMethodExplicit=DOMAIN, queue=None,
    resultsFormat=ODB, scratch='', type=ANALYSIS, userSubroutine='', waitHours=
    0, waitMinutes=0)

# submit job
mdb.jobs[job_name].submit(consistencyChecking=OFF)

# wait for job completion
mdb.jobs[job_name].waitForCompletion()

# inputs
odb = job_name + '.odb'
step = 'Step-1'
variable = 'RF2'

# access .odb
odb = openOdb(odb)

# pull values
values = np.zeros(numEvalIntervals+1)
max_ratio_ALLAE = 0.
max_ratio_ALLSD = 0.
max_ratio_ALLKE = 0.
warning_flag = False
for iter, key in enumerate(odb.steps[step].historyRegions.keys()):
    if key == 'Assembly ASSEMBLY':
        full_data = np.array(odb.steps[step].historyRegions[key].historyOutputs['ALLAE'].data)
        if len(full_data.shape) > 0:
            energies_ALLAE = np.array(odb.steps[step].historyRegions[key].historyOutputs['ALLAE'].data)[:,1]
            energies_ALLSD = np.array(odb.steps[step].historyRegions[key].historyOutputs['ALLSD'].data)[:,1]
            energies_ALLKE = np.array(odb.steps[step].historyRegions[key].historyOutputs['ALLKE'].data)[:,1]
            energies_ALLIE = np.array(odb.steps[step].historyRegions[key].historyOutputs['ALLIE'].data)[:,1]
            for i in range(1, len(energies_ALLAE)):
                if energies_ALLAE[i]/energies_ALLIE[i] > max_ratio_ALLAE:
                    max_ratio_ALLAE = energies_ALLAE[i]/energies_ALLIE[i]  
                if energies_ALLSD[i]/energies_ALLIE[i] > max_ratio_ALLSD:
                    max_ratio_ALLSD = energies_ALLSD[i]/energies_ALLIE[i] 
                if energies_ALLKE[i]/energies_ALLIE[i] > max_ratio_ALLKE:
                    max_ratio_ALLKE = energies_ALLKE[i]/energies_ALLIE[i]

            if (max_ratio_ALLAE > 0.01 or max_ratio_ALLSD > 0.01 or max_ratio_ALLKE > 0.01):
                warning_flag = True

    if key != 'Assembly ASSEMBLY':
        full_data = np.array(odb.steps[step].historyRegions[key].historyOutputs[variable].data)
        if len(full_data.shape) > 0:
            data = np.array(odb.steps[step].historyRegions[key].historyOutputs[variable].data)[:,1]
            for i in range(len(data)):
                values[i] += data[i]

flags = np.array([warning_flag, periodic_bc_flag], dtype=np.bool_)
np.savetxt(os.path.join(save_path, 'solver_flags.csv'), flags, delimiter = ',', fmt='%i')
stress_strain_curve = np.stack((strain_spacing, -values[:-1]), axis=1)
np.savetxt(os.path.join(save_path, 'stress_strain.csv'), stress_strain_curve, delimiter = ',', comments = '', header = 'strain, stress')
np.savetxt(os.path.join(save_path, 'geometry.csv'), selected_geometry.reshape(-1), delimiter = ',')

if store_frames:

    # update pixels since we generate the topology based on 48x48 pixels but evaluate it on 96x96 pixels
    # diffusion model operates on 96x96 pixels but we only consider a quarter to ensure symmetry and since we observed more variety in stress-strain responses, also makes periodic boundaries easier
    pixels = cad_pixels

    # This required some hacks since Abaqus unfortunately makes it quite hard to access values on a pixel grid via session.XYDataFromPath.
    # For reasons beyond my comprehension, it seems like we can only access two field variables (including coordinates) with each call to session.XYDataFromPath.
    # This should not be an issue since we can just call session.XYDataFromPath multiple times, but it turns out that these calls are not consistent and give different pixel arrangements (likely a bug within Abaqus)!
    # To bypass this, we leverage the fact that every negative jump in y-coordinates corresponds to a new x-coordinate, and hence can extract position and field variable with a single call to session.XYDataFromPath.

    # We then call session.XYDataFromPath three times for each variable to to also capture pixels that have left the UC to the left or right (which we map back into it).

    # this is based on the assumption that every negative jump of y-displacements
    # corresponds to a new x-coordinate, only very weird meshes would break this,
    # which we also check by verifying the the x-coordinate jumps are equal to the pixels
    
    def add_x_coordinates(data, x_pixels, x_coords, check=True):
        data = np.insert(data, 0, 0., axis=1)
        x_coord_column = 0
        y_coord_prev = 0.
        for j in range(data.shape[0]):
            y_coord = data[j,1]
            if y_coord < y_coord_prev:
                x_coord_column += 1
            data[j,0] = x_coords[x_coord_column]
            y_coord_prev = y_coord
        if check:
            if x_coord_column != x_pixels - 1:
                raise ValueError('There was a jump in the coordinates, mesh must be very weird.')
        return data

    def coords_to_pixel_index(x, y, pixels, domain_length, shift):
        x_shifted, y_shifted = x - shift, y - shift
        x_pixel = np.rint((x_shifted / domain_length)*pixels)
        y_pixel = np.rint((y_shifted / domain_length)*pixels)
        # NOTE reverse y-axis as numpy/torch go from top to bottom
        y_pixel = pixels-1 - y_pixel
        # indices = np.array([x_pixel*pixels + y_pixel], dtype=np.int64)
        indices = np.array([x_pixel + y_pixel*pixels], dtype=np.int64)
        return indices.flatten()

    session.viewports['Viewport: 1'].setValues(displayedObject=odb)

    num_frames = 11

    # create tuple of coordinates
    coords = []
    # span eval space
    x_eval = np.linspace(0., domain_length, num = pixels + 1)
    y_eval = np.linspace(0., domain_length, num = pixels + 1)
    shift = (domain_length / pixels) / 2.
    x_eval = x_eval[0:-1] + shift
    y_eval = y_eval[0:-1] + shift
    for x in x_eval:
        for y in y_eval:
            coords.append((x, y, 0.))
    session.Path(name='pixel_grid', type=POINT_LIST, expression=(tuple(coords)))

    # create tuple of coordinates for left side of UC
    coords_ls = []
    # span eval space
    x_eval_ls = np.linspace(0., -domain_length, num = pixels + 1)
    # note that we here need to shift the x-coordinates to the left
    x_eval_ls = x_eval_ls[0:-1] - shift
    for x in x_eval_ls:
        for y in y_eval:
            coords_ls.append((x, y, 0.))
    session.Path(name='pixel_grid_ls', type=POINT_LIST, expression=(tuple(coords_ls)))

    # create tuple of coordinates for right side of UC
    coords_rs = []
    # span eval space
    x_eval_rs = np.linspace(domain_length, 2*domain_length, num = pixels + 1)
    x_eval_rs = x_eval_rs[0:-1] + shift
    for x in x_eval_rs:
        for y in y_eval:
            coords_rs.append((x, y, 0.))
    session.Path(name='pixel_grid_rs', type=POINT_LIST, expression=(tuple(coords_rs)))

    geom_full_eul = np.zeros((num_frames, pixels * pixels), dtype=np.int32)
    von_Mises_full_eul = np.zeros((num_frames, pixels * pixels))
    S_y_full_eul = np.zeros((num_frames, pixels * pixels))
    strain_energy_dens_full_eul = np.zeros((num_frames, pixels * pixels))

    geom_full_lagr = np.zeros((num_frames, pixels * pixels), dtype=np.int32)
    x_disp_full_lagr = np.zeros((num_frames, pixels * pixels))
    y_disp_full_lagr = np.zeros((num_frames, pixels * pixels))
    von_Mises_full_lagr = np.zeros((num_frames, pixels * pixels))
    S_y_full_lagr = np.zeros((num_frames, pixels * pixels))
    strain_energy_dens_full_lagr = np.zeros((num_frames, pixels * pixels))

    for frame in range(0, num_frames):
        # Eulerian frame
        # von-Mises stress
        # if XYData cannot be accessed, simulation did not go through - abort here
        try:
            session.XYDataFromPath(name='von_Mises',
                removeDuplicateXYPairs=True,
                viewport = 'Viewport: 1',
                path=session.paths['pixel_grid'],
                includeIntersections=False,
                shape=DEFORMED,
                labelType=Y_COORDINATE,
                variable=(
                    ('S',INTEGRATION_POINT, ((INVARIANT, 'Mises' ), ), )
                    ),
                frame=frame+1)
        except:
            break

        von_Mises = np.asarray(session.xyDataObjects['von_Mises'].data)

        # add x coordinate based on ascending y assumption
        von_Mises = add_x_coordinates(von_Mises, pixels, x_eval)

        # convert coordinates to pixel index
        idcs = coords_to_pixel_index(von_Mises[:,0], von_Mises[:,1], pixels, domain_length, shift)

        # remove duplicates from idcs and x_coord, y_coord, raise Warning if duplicates are found
        idcs, idx_uniq = np.unique(idcs, return_index=True)
        if len(idcs) != len(von_Mises[:,0]):
            Warning('Non-unique indices found. This might especially be due to circular shift. We will continue with the first occurence.')

        # we sort the indices to avoid the sorting of np.unique, since this might be an issue for inconsistent orderings
        von_Mises_red = von_Mises[np.sort(idx_uniq),:]
        idcs = coords_to_pixel_index(von_Mises_red[:,0], von_Mises_red[:,1], pixels, domain_length, shift)
        von_Mises_full_eul[frame, idcs] = von_Mises_red[:,2]

        # add topology
        geom_full_eul[frame, idcs] = 1

        try:
            # add left side of UC
            session.XYDataFromPath(name='von_Mises_ls',
                removeDuplicateXYPairs=True,
                viewport = 'Viewport: 1',
                path=session.paths['pixel_grid_ls'],
                includeIntersections=False,
                shape=DEFORMED,
                labelType=Y_COORDINATE,
                variable=(
                    ('S',INTEGRATION_POINT, ((INVARIANT, 'Mises' ), ), )
                    ),
                frame=frame+1)

            von_Mises_ls = np.asarray(session.xyDataObjects['von_Mises_ls'].data)

            # add x coordinate based on ascending y assumption
            # NOTE: that we reversed the order of the x coordinates above for this particular instance
            von_Mises_ls = add_x_coordinates(von_Mises_ls, pixels, x_eval_ls, check=False)

            # shift coordinates to UC
            von_Mises_ls[:,0] = von_Mises_ls[:,0] + domain_length

            # convert coordinates to pixel index
            idcs = coords_to_pixel_index(von_Mises_ls[:,0], von_Mises_ls[:,1], pixels, domain_length, shift)

            # remove duplicates from idcs and x_coord, y_coord, raise Warning if duplicates are found
            idcs, idx_uniq = np.unique(idcs, return_index=True)
            if len(idcs) != len(von_Mises_ls[:,0]):
                Warning('Non-unique indices found.')

            # we sort the indices to avoid the sorting of np.unique, since this might be an issue for inconsistent orderings
            von_Mises_ls_red = von_Mises_ls[np.sort(idx_uniq),:]
            idcs = coords_to_pixel_index(von_Mises_ls_red[:,0], von_Mises_ls_red[:,1], pixels, domain_length, shift)
            # NOTE: we overwrite the values since sometimes we get duplicates, this is of course a slight inaccuracy but probably negligible
            # Another option would be to average the values or take the first occurence.
            von_Mises_full_eul[frame, idcs] = von_Mises_ls_red[:,2]

            # add topology
            geom_full_eul[frame, idcs] = 1

        except:
            pass # Bad practice, I know.

        try:
            # add right side of UC
            session.XYDataFromPath(name='von_Mises_rs',
                removeDuplicateXYPairs=True,
                viewport = 'Viewport: 1',
                path=session.paths['pixel_grid_rs'],
                includeIntersections=False,
                shape=DEFORMED,
                labelType=Y_COORDINATE,
                variable=(
                    ('S',INTEGRATION_POINT, ((INVARIANT, 'Mises' ), ), )
                    ),
                frame=frame+1)

            von_Mises_rs = np.asarray(session.xyDataObjects['von_Mises_rs'].data)

            # add x coordinate based on ascending y assumption
            von_Mises_rs = add_x_coordinates(von_Mises_rs, pixels, x_eval_rs, check=False)

            # shift coordinates to UC
            von_Mises_rs[:,0] = von_Mises_rs[:,0] - domain_length

            # convert coordinates to pixel index
            idcs = coords_to_pixel_index(von_Mises_rs[:,0], von_Mises_rs[:,1], pixels, domain_length, shift)

            # remove duplicates from idcs and x_coord, y_coord, raise Warning if duplicates are found
            idcs, idx_uniq = np.unique(idcs, return_index=True)
            if len(idcs) != len(von_Mises_rs[:,0]):
                Warning('Non-unique indices found.')

            # we sort the indices to avoid the sorting of np.unique, since this might be an issue for inconsistent orderings
            von_Mises_rs_red = von_Mises_rs[np.sort(idx_uniq),:]
            idcs = coords_to_pixel_index(von_Mises_rs_red[:,0], von_Mises_rs_red[:,1], pixels, domain_length, shift)
            von_Mises_full_eul[frame, idcs] = von_Mises_rs_red[:,2]
            
            # add topology
            geom_full_eul[frame, idcs] = 1            
        except:
            pass

        # stress in y-direction
        session.XYDataFromPath(name='S_y',
            removeDuplicateXYPairs=True,
            viewport = 'Viewport: 1',
            path=session.paths['pixel_grid'],
            includeIntersections=False,
            shape=DEFORMED,
            labelType=Y_COORDINATE,
            variable=(
                ('S',INTEGRATION_POINT, ((COMPONENT, 'S22' ), ), )
                ),
            frame=frame+1)

        S_y = np.asarray(session.xyDataObjects['S_y'].data)

        # add x coordinate based on ascending y assumption
        S_y = add_x_coordinates(S_y, pixels, x_eval)
        # convert coordinates to pixel index
        idcs = coords_to_pixel_index(S_y[:,0], S_y[:,1], pixels, domain_length, shift)
        # remove duplicates from idcs and x_coord, y_coord, raise Warning if duplicates are found
        idcs, idx_uniq = np.unique(idcs, return_index=True)
        if len(idcs) != len(S_y[:,0]):
            Warning('Non-unique indices found.')
        # we sort the indices to avoid the sorting of np.unique, since this might be an issue for inconsistent orderings
        S_y_red = S_y[np.sort(idx_uniq),:]
        idcs = coords_to_pixel_index(S_y_red[:,0], S_y_red[:,1], pixels, domain_length, shift)
        S_y_full_eul[frame, idcs] = S_y_red[:,2]

        try:
            # add left side of UC
            session.XYDataFromPath(name='S_y_ls',
                removeDuplicateXYPairs=True,
                viewport = 'Viewport: 1',
                path=session.paths['pixel_grid_ls'],
                includeIntersections=False,
                shape=DEFORMED,
                labelType=Y_COORDINATE,
                variable=(
                    ('S',INTEGRATION_POINT, ((COMPONENT, 'S22' ), ), )
                    ),
                frame=frame+1)

            S_y_ls = np.asarray(session.xyDataObjects['S_y_ls'].data)

            # add x coordinate based on ascending y assumption
            # NOTE: that we reversed the order of the x coordinates above for this particular instance
            S_y_ls = add_x_coordinates(S_y_ls, pixels, x_eval_ls, check=False)

            # shift coordinates to UC
            S_y_ls[:,0] = S_y_ls[:,0] + domain_length

            # convert coordinates to pixel index
            idcs = coords_to_pixel_index(S_y_ls[:,0], S_y_ls[:,1], pixels, domain_length, shift)

            # remove duplicates from idcs and x_coord, y_coord, raise Warning if duplicates are found
            idcs, idx_uniq = np.unique(idcs, return_index=True)
            if len(idcs) != len(S_y_ls[:,0]):
                Warning('Non-unique indices found.')

            # we sort the indices to avoid the sorting of np.unique, since this might be an issue for inconsistent orderings
            S_y_ls_red = S_y_ls[np.sort(idx_uniq),:]
            idcs = coords_to_pixel_index(S_y_ls_red[:,0], S_y_ls_red[:,1], pixels, domain_length, shift)
            S_y_full_eul[frame, idcs] = S_y_ls_red[:,2]

        except:
            pass

        try:
            # add right side of UC
            session.XYDataFromPath(name='S_y_rs',
                removeDuplicateXYPairs=True,
                viewport = 'Viewport: 1',
                path=session.paths['pixel_grid_rs'],
                includeIntersections=False,
                shape=DEFORMED,
                labelType=Y_COORDINATE,
                variable=(
                    ('S',INTEGRATION_POINT, ((COMPONENT, 'S22' ), ), )
                    ),
                frame=frame+1)

            S_y_rs = np.asarray(session.xyDataObjects['S_y_rs'].data)

            # add x coordinate based on ascending y assumption
            S_y_rs = add_x_coordinates(S_y_rs, pixels, x_eval_rs, check=False)

            # shift coordinates to UC
            S_y_rs[:,0] = S_y_rs[:,0] - domain_length

            # convert coordinates to pixel index
            idcs = coords_to_pixel_index(S_y_rs[:,0], S_y_rs[:,1], pixels, domain_length, shift)

            # remove duplicates from idcs and x_coord, y_coord, raise Warning if duplicates are found
            idcs, idx_uniq = np.unique(idcs, return_index=True)
            if len(idcs) != len(S_y_rs[:,0]):
                Warning('Non-unique indices found.')

            # we sort the indices to avoid the sorting of np.unique, since this might be an issue for inconsistent orderings
            S_y_rs_red = S_y_rs[np.sort(idx_uniq),:]
            idcs = coords_to_pixel_index(S_y_rs_red[:,0], S_y_rs_red[:,1], pixels, domain_length, shift)
            S_y_full_eul[frame, idcs] = S_y_rs_red[:,2]
                     
        except:
            pass

        # strain energy
        session.XYDataFromPath(name='energy',
            removeDuplicateXYPairs=True,
            viewport = 'Viewport: 1',
            path=session.paths['pixel_grid'],
            includeIntersections=False,
            shape=DEFORMED,
            labelType=Y_COORDINATE,
            variable=(
                ('SENER', INTEGRATION_POINT),
                ),
            frame=frame+1)

        energy = np.asarray(session.xyDataObjects['energy'].data)
        # add x coordinate based on ascending y assumption
        energy = add_x_coordinates(energy, pixels, x_eval)
        # convert coordinates to pixel index
        idcs = coords_to_pixel_index(energy[:,0], energy[:,1], pixels, domain_length, shift)
        # remove duplicates from idcs and x_coord, y_coord, raise Warning if duplicates are found
        idcs, idx_uniq = np.unique(idcs, return_index=True)
        if len(idcs) != len(energy[:,0]):
            Warning('Non-unique indices found.')
        # we sort the indices to avoid the sorting of np.unique, since this might be an issue for inconsistent orderings
        energy_red = energy[np.sort(idx_uniq),:]
        idcs = coords_to_pixel_index(energy_red[:,0], energy_red[:,1], pixels, domain_length, shift)
        strain_energy_dens_full_eul[frame, idcs] = energy_red[:,2]

        try:
            # add left side of UC
            session.XYDataFromPath(name='energy_ls',
                removeDuplicateXYPairs=True,
                viewport = 'Viewport: 1',
                path=session.paths['pixel_grid_ls'],
                includeIntersections=False,
                shape=DEFORMED,
                labelType=Y_COORDINATE,
                variable=(
                    ('SENER', INTEGRATION_POINT),
                    ),
                frame=frame+1)

            energy_ls = np.asarray(session.xyDataObjects['energy_ls'].data)

            # add x coordinate based on ascending y assumption
            # NOTE: that we reversed the order of the x coordinates above for this particular instance
            energy_ls = add_x_coordinates(energy_ls, pixels, x_eval_ls, check=False)

            # shift coordinates to UC
            energy_ls[:,0] = energy_ls[:,0] + domain_length

            # convert coordinates to pixel index
            idcs = coords_to_pixel_index(energy_ls[:,0], energy_ls[:,1], pixels, domain_length, shift)

            # remove duplicates from idcs and x_coord, y_coord, raise Warning if duplicates are found
            idcs, idx_uniq = np.unique(idcs, return_index=True)
            if len(idcs) != len(energy_ls[:,0]):
                Warning('Non-unique indices found.')

            # we sort the indices to avoid the sorting of np.unique, since this might be an issue for inconsistent orderings
            energy_ls_red = energy_ls[np.sort(idx_uniq),:]
            idcs = coords_to_pixel_index(energy_ls_red[:,0], energy_ls_red[:,1], pixels, domain_length, shift)
            strain_energy_dens_full_eul[frame, idcs] = energy_ls_red[:,2]

        except:
            pass

        try:
            # add right side of UC
            session.XYDataFromPath(name='energy_rs',
                removeDuplicateXYPairs=True,
                viewport = 'Viewport: 1',
                path=session.paths['pixel_grid_rs'],
                includeIntersections=False,
                shape=DEFORMED,
                labelType=Y_COORDINATE,
                variable=(
                    ('SENER', INTEGRATION_POINT),
                    ),
                frame=frame+1)

            energy_rs = np.asarray(session.xyDataObjects['energy_rs'].data)

            # add x coordinate based on ascending y assumption
            energy_rs = add_x_coordinates(energy_rs, pixels, x_eval_rs, check=False)

            # shift coordinates to UC
            energy_rs[:,0] = energy_rs[:,0] - domain_length

            # convert coordinates to pixel index
            idcs = coords_to_pixel_index(energy_rs[:,0], energy_rs[:,1], pixels, domain_length, shift)

            # remove duplicates from idcs and x_coord, y_coord, raise Warning if duplicates are found
            idcs, idx_uniq = np.unique(idcs, return_index=True)
            if len(idcs) != len(energy_rs[:,0]):
                Warning('Non-unique indices found.')

            # we sort the indices to avoid the sorting of np.unique, since this might be an issue for inconsistent orderings
            energy_rs_red = energy_rs[np.sort(idx_uniq),:]
            idcs = coords_to_pixel_index(energy_rs_red[:,0], energy_rs_red[:,1], pixels, domain_length, shift)
            strain_energy_dens_full_eul[frame, idcs] = energy_rs_red[:,2]
                
        except:
            pass

        # Lagrangian frame
        # x-displacement
        # if XYData cannot be accessed, simulation did not go through - abort here
        try:
            session.XYDataFromPath(name='x_disp',
                removeDuplicateXYPairs=True,
                viewport = 'Viewport: 1',
                path=session.paths['pixel_grid'],
                includeIntersections=False,
                shape=UNDEFORMED,
                labelType=Y_COORDINATE,
                variable=(
                    ('U', NODAL, ((COMPONENT, 'U1' ), ), )
                    ),
                frame=frame+1)
        except:
            break

        x_disp = np.asarray(session.xyDataObjects['x_disp'].data)

        # add x coordinate based on ascending y assumption
        x_disp = add_x_coordinates(x_disp, pixels, x_eval)
        # convert coordinates to pixel index
        idcs = coords_to_pixel_index(x_disp[:,0], x_disp[:,1], pixels, domain_length, shift)
        # remove duplicates from idcs and x_coord, y_coord, raise Warning if duplicates are found
        idcs, idx_uniq = np.unique(idcs, return_index=True)
        if len(idcs) != len(x_disp[:,0]):
            Warning('Non-unique indices found.')
        # we sort the indices to avoid the sorting of np.unique, since this might be an issue for inconsistent orderings
        x_disp_red = x_disp[np.sort(idx_uniq),:]
        idcs = coords_to_pixel_index(x_disp_red[:,0], x_disp_red[:,1], pixels, domain_length, shift)
        x_disp_full_lagr[frame, idcs] = x_disp_red[:,2]

        # add topology
        geom_full_lagr[frame, idcs] = 1

        # y-displacement
        # if XYData cannot be accessed, simulation did not go through - abort here
        try:
            session.XYDataFromPath(name='y_disp',
                removeDuplicateXYPairs=True,
                viewport = 'Viewport: 1',
                path=session.paths['pixel_grid'],
                includeIntersections=False,
                shape=UNDEFORMED,
                labelType=Y_COORDINATE,
                variable=(
                    ('U', NODAL, ((COMPONENT, 'U2' ), ), )
                    ),
                frame=frame+1)
        except:
            break

        y_disp = np.asarray(session.xyDataObjects['y_disp'].data)

        # add x coordinate based on ascending y assumption
        y_disp = add_x_coordinates(y_disp, pixels, x_eval)
        # convert coordinates to pixel index
        idcs = coords_to_pixel_index(y_disp[:,0], y_disp[:,1], pixels, domain_length, shift)
        # remove duplicates from idcs and x_coord, y_coord, raise Warning if duplicates are found
        idcs, idx_uniq = np.unique(idcs, return_index=True)
        if len(idcs) != len(y_disp[:,0]):
            Warning('Non-unique indices found.')
        # we sort the indices to avoid the sorting of np.unique, since this might be an issue for inconsistent orderings
        y_disp_red = y_disp[np.sort(idx_uniq),:]
        idcs = coords_to_pixel_index(y_disp_red[:,0], y_disp_red[:,1], pixels, domain_length, shift)
        y_disp_full_lagr[frame, idcs] = y_disp_red[:,2]

        # von-Mises stress
        # if XYData cannot be accessed, simulation did not go through - abort here
        try:
            session.XYDataFromPath(name='von_Mises',
                removeDuplicateXYPairs=True,
                viewport = 'Viewport: 1',
                path=session.paths['pixel_grid'],
                includeIntersections=False,
                shape=UNDEFORMED,
                labelType=Y_COORDINATE,
                variable=(
                    ('S',INTEGRATION_POINT, ((INVARIANT, 'Mises' ), ), )
                    ),
                frame=frame+1)
        except:
            break

        von_Mises = np.asarray(session.xyDataObjects['von_Mises'].data)

        # add x coordinate based on ascending y assumption
        von_Mises = add_x_coordinates(von_Mises, pixels, x_eval)
        # convert coordinates to pixel index
        idcs = coords_to_pixel_index(von_Mises[:,0], von_Mises[:,1], pixels, domain_length, shift)
        # remove duplicates from idcs and x_coord, y_coord, raise Warning if duplicates are found
        idcs, idx_uniq = np.unique(idcs, return_index=True)
        if len(idcs) != len(von_Mises[:,0]):
            Warning('Non-unique indices found.')
        # we sort the indices to avoid the sorting of np.unique, since this might be an issue for inconsistent orderings
        von_Mises_red = von_Mises[np.sort(idx_uniq),:]
        idcs = coords_to_pixel_index(von_Mises_red[:,0], von_Mises_red[:,1], pixels, domain_length, shift)
        von_Mises_full_lagr[frame, idcs] = von_Mises_red[:,2]

        # stress in y-direction
        session.XYDataFromPath(name='S_y',
            removeDuplicateXYPairs=True,
            viewport = 'Viewport: 1',
            path=session.paths['pixel_grid'],
            includeIntersections=False,
            shape=UNDEFORMED,
            labelType=Y_COORDINATE,
            variable=(
                ('S',INTEGRATION_POINT, ((COMPONENT, 'S22' ), ), )
                ),
            frame=frame+1)

        S_y = np.asarray(session.xyDataObjects['S_y'].data)

        # add x coordinate based on ascending y assumption
        S_y = add_x_coordinates(S_y, pixels, x_eval)
        # convert coordinates to pixel index
        idcs = coords_to_pixel_index(S_y[:,0], S_y[:,1], pixels, domain_length, shift)
        # remove duplicates from idcs and x_coord, y_coord, raise Warning if duplicates are found
        idcs, idx_uniq = np.unique(idcs, return_index=True)
        if len(idcs) != len(S_y[:,0]):
            Warning('Non-unique indices found.')
        # we sort the indices to avoid the sorting of np.unique, since this might be an issue for inconsistent orderings
        S_y_red = S_y[np.sort(idx_uniq),:]
        idcs = coords_to_pixel_index(S_y_red[:,0], S_y_red[:,1], pixels, domain_length, shift)
        S_y_full_lagr[frame, idcs] = S_y_red[:,2]

        # strain energy
        session.XYDataFromPath(name='energy',
            removeDuplicateXYPairs=True,
            viewport = 'Viewport: 1',
            path=session.paths['pixel_grid'],
            includeIntersections=False,
            shape=UNDEFORMED,
            labelType=Y_COORDINATE,
            variable=(
                ('SENER', INTEGRATION_POINT),
                ),
            frame=frame+1)

        energy = np.asarray(session.xyDataObjects['energy'].data)

        # add x coordinate based on ascending y assumption
        energy = add_x_coordinates(energy, pixels, x_eval)
        # convert coordinates to pixel index
        idcs = coords_to_pixel_index(energy[:,0], energy[:,1], pixels, domain_length, shift)
        # remove duplicates from idcs and x_coord, y_coord, raise Warning if duplicates are found
        idcs, idx_uniq = np.unique(idcs, return_index=True)
        if len(idcs) != len(energy[:,0]):
            Warning('Non-unique indices found.')
        # we sort the indices to avoid the sorting of np.unique, since this might be an issue for inconsistent orderings
        energy_red = energy[np.sort(idx_uniq),:]
        idcs = coords_to_pixel_index(energy_red[:,0], energy_red[:,1], pixels, domain_length, shift)
        strain_energy_dens_full_lagr[frame, idcs] = energy_red[:,2]

    # save data also as csv for further post-processing
    np.savetxt(os.path.join(save_path, 'geometry_frames_eul.csv'), geom_full_eul, delimiter = ',', fmt='%i')
    np.savetxt(os.path.join(save_path, 'von_Mises_frames_eul.csv'), von_Mises_full_eul, delimiter = ',')
    np.savetxt(os.path.join(save_path, 's_22_frames_eul.csv'), S_y_full_eul, delimiter = ',')
    np.savetxt(os.path.join(save_path, 'strain_energy_dens_frames_eul.csv'), strain_energy_dens_full_eul, delimiter = ',')

    np.savetxt(os.path.join(save_path, 'geometry_frames_lagr.csv'), geom_full_lagr, delimiter = ',', fmt='%i')
    np.savetxt(os.path.join(save_path, 'x_disp_frames_lagr.csv'), x_disp_full_lagr, delimiter = ',')
    np.savetxt(os.path.join(save_path, 'y_disp_frames_lagr.csv'), y_disp_full_lagr, delimiter = ',')
    np.savetxt(os.path.join(save_path, 'von_Mises_frames_lagr.csv'), von_Mises_full_lagr, delimiter = ',')
    np.savetxt(os.path.join(save_path, 's_22_frames_lagr.csv'), S_y_full_lagr, delimiter = ',')
    np.savetxt(os.path.join(save_path, 'strain_energy_dens_frames_lagr.csv'), strain_energy_dens_full_lagr, delimiter = ',')

odb.close()