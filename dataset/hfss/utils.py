"""
HFSS Utilities Module
Contains core functions for HFSS project manipulation and analysis
"""
import win32com.client
import pythoncom

# HFSS Application Constants
UNITS = [
	"NAME:Units Parameter",
	"Units:=", "um",
	"Rescale:=", False
]


def new_project(scale, desktop, design_name, material, copper_height, material_height):
	"""
	Create new HFSS project with base configuration
	Returns:
		win32com.client.CDispatch: HFSS Project object
	"""
	project = desktop.NewProject()
	project.InsertDesign("HFSS", design_name, "DrivenModal", "")
	
	design = project.SetActiveDesign(design_name)
	editor = design.SetActiveEditor("3D Modeler")
	editor.SetModelUnits(UNITS)
	
	# Configure design parameters
	design.ChangeProperty( [
		"NAME:AllTabs",
		[
			"NAME:LocalVariableTab",
			[
				"NAME:PropServers", 
				"LocalVariables"
			],
			[
				"NAME:NewProps",
				[   # unit size
					"NAME:side",	
					"PropType:=", "VariableProp",
					"UserDef:=", True,
					"Value:=", str(scale) + "um"
				],
				[   # height
					"NAME:height",
					"PropType:=", "VariableProp",
					"UserDef:=", True,
					"Value:=", str(material_height) + "um"
				],
			]
		]
	])
	
	# Create airbox
	z_length = max(64 * scale, 4 * (copper_height + material_height))
	editor.CreateBox(
	[
		"NAME:BoxParameters",
		"XPosition:="		, "0um",
		"YPosition:="		, "0um",
		"ZPosition:="		, "-" + str(z_length/2) + "um",
		"XSize:="		, str(64 * scale) + "um",
		"YSize:="		, str(64 * scale) + "um",
		"ZSize:="		, str(z_length) + "um"
	], 
	[
		"NAME:Attributes",
		"Name:="		, "AirBox",
		"Flags:="		, "",
		"Color:="		, "(0 128 255)",
		"Transparency:="	, 0.8,
		"PartCoordinateSystem:=", "Global",
		"UDMId:="		, "",
		"MaterialValue:="	, "\"vacuum\"",
		"SurfaceMaterialValue:=", "\"\"",
		"SolveInside:="		, True,
		"IsMaterialEditable:="	, True,
		"UseMaterialAppearance:=", False,
		"IsLightweight:="	, False
	])
	Module = design.GetModule("BoundarySetup")

	# Create ports 
	editor.CreateRectangle(
	[
		"NAME:RectangleParameters",
		"IsCovered:="		, True,
		"XStart:="		, "0um",
		"YStart:="		, "0um",
		"ZStart:="		, "-" + str(z_length/2) + "um",
		"Width:="		, str(z_length) + "um",
		"Height:="		, str(64 * scale) + "um",
		"WhichAxis:="		, "Y"
	], 
	[
		"NAME:Attributes",
		"Name:="		, "Port1",
		"Flags:="		, "",
		"Color:="		, "(255 0 0)",
		"Transparency:="	, 0.9,
		"PartCoordinateSystem:=", "Global",
		"UDMId:="		, "",
		"MaterialValue:="	, "\"vacuum\"",
		"SurfaceMaterialValue:=", "\"\"",
		"SolveInside:="		, True,
		"IsMaterialEditable:="	, True,
		"UseMaterialAppearance:=", False,
		"IsLightweight:="	, False
	])
	editor.CreateRectangle(
	[
		"NAME:RectangleParameters",
		"IsCovered:="		, True,
		"XStart:="		, "0um",
		"YStart:="		, str(64 * scale) + "um",
		"ZStart:="		, "-" + str(z_length/2) + "um",
		"Width:="		, str(z_length) + "um",
		"Height:="		, str(64 * scale) + "um",
		"WhichAxis:="		, "Y"
	], 
	[
		"NAME:Attributes",
		"Name:="		, "Port2",
		"Flags:="		, "",
		"Color:="		, "(255 0 0)",
		"Transparency:="	, 0.9,
		"PartCoordinateSystem:=", "Global",
		"UDMId:="		, "",
		"MaterialValue:="	, "\"vacuum\"",
		"SurfaceMaterialValue:=", "\"\"",
		"SolveInside:="		, True,
		"IsMaterialEditable:="	, True,
		"UseMaterialAppearance:=", False,
		"IsLightweight:="	, False
	])
	
	Module.AssignPerfectE(
	[
		"NAME:PerfE1",
		"Faces:="		, [10],
		"InfGroundPlane:="	, False
	])
	Module.AssignPerfectE(
	[
		"NAME:PerfE2",
		"Faces:="		, [12],
		"InfGroundPlane:="	, False
	])
	Module.AssignPerfectH(
	[
		"NAME:PerfH1",
		"Faces:="		, [7],
		"InfGroundPlane:="	, False
	])
	Module.AssignPerfectH(
	[
		"NAME:PerfH2",
		"Faces:="		, [8],
		"InfGroundPlane:="	, False
	])
	# Create dielectric substrate
	editor.CreateBox(
	[
		"NAME:BoxParameters",
		"XPosition:="		, "0um",
		"YPosition:="		, "0um",
		"ZPosition:="		, "-" + "height",
		"XSize:="		, str(64 * scale) + "um",
		"YSize:="		, str(64 * scale) + "um",
		"ZSize:="		, "height"
	], 
	[
		"NAME:Attributes",
		"Name:="		, "Substrate",
		"Flags:="		, "",
		"Color:="		, "(128 128 128)",
		"Transparency:="	, 0,
		"PartCoordinateSystem:=", "Global",
		"UDMId:="		, "",
		"MaterialValue:="	, material,
		"SurfaceMaterialValue:=", "\"\"",
		"SolveInside:="		, True,
		"IsMaterialEditable:="	, True,
		"UseMaterialAppearance:=", False,
		"IsLightweight:="	, False
	])
	Module.AssignWavePort(
	[
		"NAME:1",
		"Objects:="		, ["Port1"],
		"NumModes:="		, 1,
		"UseLineModeAlignment:=", False,
		"DoDeembed:="		, False,
		"RenormalizeAllTerminals:=", True,
		[
			"NAME:Modes",
			[
				"NAME:Mode1",
				"ModeNum:="		, 1,
				"UseIntLine:="		, True,
				[
					"NAME:IntLine",
					"Start:="		, ["0um","0um","0um"],
					"End:="			, [str(64 * scale) + "um", "0um","0um"]
				],
				"AlignmentGroup:="	, 0,
				"CharImp:="		, "Zpi",
				"RenormImp:="		, "50ohm"
			]
		],
		"ShowReporterFilter:="	, False,
		"ReporterFilter:="	, [True],
		"UseAnalyticAlignment:=", False
	])
	Module.AssignWavePort(
	[
		"NAME:2",
		"Objects:="		, ["Port2"],
		"NumModes:="		, 1,
		"UseLineModeAlignment:=", False,
		"DoDeembed:="		, False,
		"RenormalizeAllTerminals:=", True,
		[
			"NAME:Modes",
			[
				"NAME:Mode1",
				"ModeNum:="		, 1,
				"UseIntLine:="		, True,
				[
					"NAME:IntLine",
					"Start:="		, ["0um",str(64 * scale) + "um","0um"],
					"End:="			, [str(64 * scale) + "um",str(64 * scale) + "um","0um"]
				],
				"AlignmentGroup:="	, 0,
				"CharImp:="		, "Zpi",
				"RenormImp:="		, "50ohm"
			]
		],
		"ShowReporterFilter:="	, False,
		"ReporterFilter:="	, [True],
		"UseAnalyticAlignment:=", False
	])
	
	return project


def get_ractangle(bitmap, method):
	"""
	Extract rectangles from binary bitmap
	"""
	rectangles = {}
	rect_id = 0
	
	# Simple method: loop through each row and extract continuous pixels as a rectangle
	for y, row in enumerate(bitmap):
		x = 0
		while x < len(row):
			if row[x] == 1:
				start_x = x
				while x < len(row) and row[x] == 1:
					x += 1
				rectangles[rect_id] = [{
					'x': start_x,
					'y': y,
					'width': x - start_x,
					'height': 1
				}]
				rect_id += 1
			else:
				x += 1
				
	return rectangles


def analyze(project_name, args, face=1, ver="2022.2"):
	"""
	Execute HFSS simulation and export results
	Args:
		project_name: Name of .aedt file to analyze
		face: Single/Double face configuration
		ver: HFSS version identifier
	"""
	pythoncom.CoInitialize()
	
	try:
		hfss = win32com.client.Dispatch(f'Ansoft.ElectronicsDesktop.{ver}')
		desktop = hfss.GetAppDesktop()
		desktop.OpenProject(f"{args.simualtion_dir}{project_name}")
		name = project_name[:-5]
		
		# set active project
		project = desktop.SetActiveProject(name)
		design_name = name.split("_")[0]
		Design = project.SetActiveDesign(design_name)
		Module = Design.GetModule("AnalysisSetup")
		Module.EditFrequencySweep("Setup1", "Sweep", 
		[
			"NAME:Sweep",
			"IsEnabled:="		, True,
			"RangeType:="		, "LinearCount",
			"RangeStart:="		, "60GHz",
			"RangeEnd:="		, "80GHz",
			"RangeCount:="		, 200,
			"Type:="		, "Fast",
			"SaveFields:="		, True,
			"SaveRadFields:="	, False,
			"GenerateFieldsForAllFreqs:=", False
		])
		
		# begin simulation
		Design.AnalyzeAll()
		Module = Design.GetModule("ReportSetup")
		if face == 1:
			Module.ExportToFile("S Parameter Plot 1", "{args.result_dir}S11_im/" + name + ".csv", False)
			Module.ExportToFile("S Parameter Plot 2", "{args.result_dir}S11_re/" + name + ".csv", False)
			Module.ExportToFile("S Parameter Plot 3", "{args.result_dir}S21_im/" + name + ".csv", False)
			Module.ExportToFile("S Parameter Plot 4", "{args.result_dir}S21_re/" + name + ".csv", False)
			
	finally:
		desktop.CloseProject(project_name[:-5])


def draw_rectangle(Editor, x, y, x_length, y_length, name, copper_height):
	"""
	Draw a rectangle in HFSS modeler
	"""
	Editor.CreateBox(
	[
		"NAME:BoxParameters",
		"XPosition:="		, str(x) + "um",
		"YPosition:="		, str(y) + "um",
		"ZPosition:="		, "0um",
		"XSize:="		, str(x_length) + "um",
		"YSize:="		, str(y_length) + "um",
		"ZSize:="		, str(copper_height) + "um",
	], 
	[
		"NAME:Attributes",
		"Name:="		, name,
		"Flags:="		, "",
		"Color:="		, "(143 175 143)",
		"Transparency:="	, 0,
		"PartCoordinateSystem:=", "Global",
		"UDMId:="		, "",
		"MaterialValue:="	, "\"copper\"",
		"SurfaceMaterialValue:=", "\"\"",
		"SolveInside:="		, False,
		"IsMaterialEditable:="	, True,
		"UseMaterialAppearance:=", False,
		"IsLightweight:="	, False
	])


def insert_mng(scale, rectangles, Project, name, copper_height):
	"""
	Insert model geometry into HFSS project and create analysis setup and analysis reports
	"""
	Design = Project.SetActiveDesign(name)
	Editor = Design.SetActiveEditor("3D Modeler")
	unite = []
	for class_id, rects in rectangles.items():
		tmp = ""
		number = 0
		for rect in rects:
			atom_name = "atom" + str(class_id) + "_" + str(number)
			tmp += atom_name  + ","
			draw_rectangle(Editor, rect['x'] * scale, rect['y'] * scale, rect['width'] * scale, rect['height'] * scale, atom_name, copper_height)
			number += 1
		if len(rects) > 1:
			unite.append(tmp[:-1])

	for item in unite:
		Editor.Unite(
		[
			"NAME:Selections",
			"Selections:="		, item
		], 
		[
			"NAME:UniteParameters",
			"KeepOriginals:="	, False
		])

    # Create analysis setup and result reports
	Module = Design.GetModule("AnalysisSetup")
	Module.InsertSetup("HfssDriven", 
	[
		"NAME:Setup1",
		"AdaptMultipleFreqs:="	, False,
		"Frequency:="		, "70GHz",
		"MaxDeltaS:="		, 0.02,
		"PortsOnly:="		, False,
		"UseMatrixConv:="	, False,
		"MaximumPasses:="	, 6,
		"MinimumPasses:="	, 1,
		"MinimumConvergedPasses:=", 1,
		"PercentRefinement:="	, 30,
		"IsEnabled:="		, True,
		[
			"NAME:MeshLink",
			"ImportMesh:="		, False
		],
		"BasisOrder:="		, 1,
		"DoLambdaRefine:="	, True,
		"DoMaterialLambda:="	, True,
		"SetLambdaTarget:="	, False,
		"Target:="		, 0.3333,
		"UseMaxTetIncrease:="	, False,
		"PortAccuracy:="	, 2,
		"UseABCOnPort:="	, False,
		"SetPortMinMaxTri:="	, False,
		"UseDomains:="		, False,
		"UseIterativeSolver:="	, False,
		"SaveRadFieldsOnly:="	, False,
		"SaveAnyFields:="	, True,
		"IESolverType:="	, "Auto",
		"LambdaTargetForIESolver:=", 0.15,
		"UseDefaultLambdaTgtForIESolver:=", True,
		"IE Solver Accuracy:="	, "Balanced"
	])
	Module.InsertFrequencySweep("Setup1", 
	[
		"NAME:Sweep",
		"IsEnabled:="		, True,
		"RangeType:="		, "LinearCount",
		"RangeStart:="		, "60GHz",
		"RangeEnd:="		, "80GHz",
		"RangeCount:="		, 200,
		"Type:="		, "Fast",
		"SaveFields:="		, True,
		"SaveRadFields:="	, False,
		"GenerateFieldsForAllFreqs:=", False
	])
	Module = Design.GetModule("ReportSetup")
	Module.CreateReport("S Parameter Plot 1", "Modal Solution Data", "Rectangular Plot", "Setup1 : Sweep", 
		[
			"Domain:="		, "Sweep"
		], 
		[
			"Freq:="		, ["All"],
			"side:="		, ["Nominal"],
			"height:="		, ["Nominal"]
		], 
		[
			"X Component:="		, "Freq",
			"Y Component:="		, ["im(S(1,1))"]
		])
	Module.CreateReport("S Parameter Plot 2", "Modal Solution Data", "Rectangular Plot", "Setup1 : Sweep", 
		[
			"Domain:="		, "Sweep"
		], 
		[
			"Freq:="		, ["All"],
			"side:="		, ["Nominal"],
			"height:="		, ["Nominal"]
		], 
		[
			"X Component:="		, "Freq",
			"Y Component:="		, ["re(S(1,1))"]
		])
	Module.CreateReport("S Parameter Plot 3", "Modal Solution Data", "Rectangular Plot", "Setup1 : Sweep", 
		[
			"Domain:="		, "Sweep"
		], 
		[
			"Freq:="		, ["All"],
			"side:="		, ["Nominal"],
			"height:="		, ["Nominal"]
		], 
		[
			"X Component:="		, "Freq",
			"Y Component:="		, ["im(S(2,1))"]
		])
	Module.CreateReport("S Parameter Plot 4", "Modal Solution Data", "Rectangular Plot", "Setup1 : Sweep", 
		[
			"Domain:="		, "Sweep"
		], 
		[
			"Freq:="		, ["All"],
			"side:="		, ["Nominal"],
			"height:="		, ["Nominal"]
		], 
		[
			"X Component:="		, "Freq",
			"Y Component:="		, ["re(S(2,1))"]
		])
	return