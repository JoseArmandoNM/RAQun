from RAQun.utils.maths import log_2, log_t

def test_log_2():
    assert log_2(1) == 1
    assert log_2(2) == 1
    assert log_2(3) == 2
    assert log_2(4) == 2
    assert log_2(5) == 3
    assert log_2(8) == 3
    assert log_2(16) == 4
#end test_log_2

def test_log_t():
    assert log_t(1) == (1, 1)
    assert log_t(2) == (2, 1)
    assert log_t(3) == (3, 2)
    assert log_t(4) == (4, 2)
    assert log_t(5) == (5, 3)
    assert log_t(8) == (8, 3)
    assert log_t(16) == (16, 4)
#end test_log_t

























