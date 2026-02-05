# VKITTI 2 colors (R, G, B)
VKITTI_2_PALETTE = {
    "Terrain": (210, 0, 200),
    "Sky": (90, 200, 255),
    "Tree": (0, 199, 0),
    "Vegetation": (90, 240, 0),
    "Building": (140, 140, 140),
    "Road": (100, 60, 100),
    "GuardRail": (250, 100, 255),
    "TrafficSign": (255, 255, 0),
    "TrafficLight": (200, 200, 0),
    "Pole": (255, 130, 0),
    "Misc": (80, 80, 80),
    "Truck": (160, 60, 60),
    "Car": (255, 127, 80),
    "Van": (0, 139, 139),
    "Undefined": (0, 0, 0),
}

# VKTTI 2 to ADE20K indices
# Index mapping used for reference: https://github.com/CSAILVision/sceneparsing/blob/master/objectInfo150.csv
VKITTI_2_TO_ADE20K_IDX = {
    "Terrain": 12,  # sidewalk;pavement
    "Sky": 3,  # sky
    "Tree": 5,  # tree
    "Vegetation": 18,  # plant;flora;plant;life
    "Building": 2,  # building;edifice
    "Road": 7,  # road;route
    "GuardRail": 33,  # fence;fencing
    "TrafficSign": 44,  # signboard;sign
    "TrafficLight": 137,  # traffic;light;traffic;signal;stoplight
    "Pole": 94,  # pole
    "Misc": 0,  # undefined
    "Truck": 84,  # truck;motortruck
    "Car": 21,  # car;auto;automobile;machine;motorcar
    "Van": 103,  # van
    "Undefined": 0,  # undefined
}
