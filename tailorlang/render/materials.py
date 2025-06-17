# Cotton Fabric BSDF
def create_cotton_bsdf():
    return {
        'type': 'blendbsdf',
        'weight': 0.7,
        'bsdf1': {
            'type': 'roughplastic',
            'alpha': 0.5,  # High roughness for cotton's matte appearance
            'diffuse_reflectance': {
                'type': 'rgb',
                'value': [0.95, 0.95, 0.95]  # Light base color
            },
            'nonlinear': True
        },
        'bsdf2': {
            'type': 'roughconductor',
            'alpha_u': 0.7,  # Highly anisotropic roughness
            'alpha_v': 0.3,
            'eta': {
                'type': 'rgb',
                'value': [0.15, 0.15, 0.15]
            },
            'k': {
                'type': 'rgb',
                'value': [0.4, 0.4, 0.4]
            }
        }
    }

# Silk Fabric BSDF
def create_silk_bsdf():
    return {
        'type': 'blendbsdf',
        'weight': 0.6,
        'bsdf1': {
            'type': 'roughplastic',
            'alpha': 0.2,  # Lower roughness for silk's sheen
            'diffuse_reflectance': {
                'type': 'rgb',
                'value': [0.98, 0.98, 0.98]  # Bright base color
            },
            'nonlinear': True
        },
        'bsdf2': {
            'type': 'roughconductor',
            'alpha_u': 0.1,  # Sharp anisotropic highlights
            'alpha_v': 0.4,
            'eta': {
                'type': 'rgb',
                'value': [0.12, 0.12, 0.12]
            },
            'k': {
                'type': 'rgb',
                'value': [0.45, 0.45, 0.45]
            }
        }
    }

# Denim Fabric BSDF
def create_denim_bsdf():
    return {
        'type': 'blendbsdf',
        'weight': 0.8,
        'bsdf1': {
            'type': 'roughplastic',
            'alpha': 0.6,  # Very rough for denim texture
            'diffuse_reflectance': {
                'type': 'rgb',
                'value': [0.2, 0.3, 0.45]  # Denim blue color
            },
            'nonlinear': True
        },
        'bsdf2': {
            'type': 'roughconductor',
            'alpha_u': 0.8,  # Strong anisotropic pattern
            'alpha_v': 0.2,
            'eta': {
                'type': 'rgb',
                'value': [0.2, 0.2, 0.2]
            },
            'k': {
                'type': 'rgb',
                'value': [0.35, 0.35, 0.35]
            }
        }
    }

# Human Skin BSDF (Caucasian)
def create_caucasian_skin_bsdf():
    return {
        'type': 'blendbsdf',
        'weight': 0.85,
        'bsdf1': {
            'type': 'diffuse',
            'reflectance': {
                'type': 'rgb',
                'value': [0.92, 0.75, 0.67]  # Base skin tone
            }
        },
        'bsdf2': {
            'type': 'dielectric',
            'int_ior': 1.4,  # Approximate IOR of skin
            'ext_ior': 1.0,
            'specular_reflectance': {
                'type': 'rgb',
                'value': [0.04, 0.04, 0.04]  # Subtle specular
            }
        }
    }

# Human Skin BSDF (Dark)
def create_dark_skin_bsdf():
    return {
        'type': 'blendbsdf',
        'weight': 0.85,
        'bsdf1': {
            'type': 'diffuse',
            'reflectance': {
                'type': 'rgb',
                'value': [0.25, 0.15, 0.1]  # Darker skin tone
            }
        },
        'bsdf2': {
            'type': 'dielectric',
            'int_ior': 1.4,
            'ext_ior': 1.0,
            'specular_reflectance': {
                'type': 'rgb',
                'value': [0.04, 0.04, 0.04]  # Similar specular properties
            }
        }
    }

# Human Skin BSDF (Asian)
def create_asian_skin_bsdf():
    return {
        'type': 'blendbsdf',
        'weight': 0.85,
        'bsdf1': {
            'type': 'diffuse',
            'reflectance': {
                'type': 'rgb',
                'value': [0.85, 0.68, 0.55]  # Medium skin tone
            }
        },
        'bsdf2': {
            'type': 'dielectric',
            'int_ior': 1.4,
            'ext_ior': 1.0,
            'specular_reflectance': {
                'type': 'rgb',
                'value': [0.04, 0.04, 0.04]
            }
        }
    }

'''
# Example usage in scene dictionary
scene_dict = {
    'type': 'scene',
    'garment': {
        'type': 'ply',
        'filename': 'garment.ply',
        'bsdf': create_cotton_bsdf()  # Or any other fabric BSDF
    },
    'body': {
        'type': 'ply',
        'filename': 'body.ply',
        'bsdf': create_caucasian_skin_bsdf()  # Or any other skin BSDF
    }
}
'''