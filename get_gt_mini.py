from waymo_open_dataset.protos.metrics_pb2 import Objects

objects = Objects()
with open('./data/waymo_mini/waymo_format/gt.bin', 'rb') as f:
    objects.ParseFromString(bytearray(f.read()))
new = Objects()
for obj in objects.objects:
    if obj.context_name == '10203656353524179475_7625_000_7645_000':
        new.objects.append(obj)
with open('./data/waymo_mini/waymo_format/gt_mini.bin', 'wb') as f:
    f.write(new.SerializeToString())
