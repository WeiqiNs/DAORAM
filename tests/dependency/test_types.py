from oblivlib.dependency import AVLData, BPlusData, Data


def test_field_tuple_order_is_the_stored_format():
    cases = [
        (Data(key=1, leaf=2, value=b"v"), [1, 2, b"v"]),
        (
            AVLData(value=b"v", r_key=b"r", r_leaf=3, r_height=1, l_key=b"l", l_leaf=4, l_height=2),
            [b"v", b"r", 3, 1, b"l", 4, 2],
        ),
        (BPlusData(keys=[b"a", b"b"], values=[[10, 0], [20, 1], [30, 2]]), [[b"a", b"b"], [[10, 0], [20, 1], [30, 2]]]),
    ]
    for obj, fields in cases:
        assert obj.to_fields() == fields
        assert type(obj).from_fields(fields) == obj
