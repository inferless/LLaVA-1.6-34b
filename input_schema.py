INPUT_SCHEMA = {
    "prompt": {
        'datatype': 'STRING',
        'required': True,
        'shape': [1],
        'example': ["Describe the scene"]
    },
    "image": {
        'datatype': 'STRING',
        'required': True,
        'shape': [1],
        'example': ["https://www.southernliving.com/thmb/IvIFZcOxtqnfGvykYNisB_xEz4I=/750x0/filters:no_upscale():max_bytes(150000):strip_icc():format(webp)/27372_LBurton_101822_05-4648d290e31e4097823387ecebc4f280.jpg"]
    }
}
