import numpy as np
import pytest
mj=pytest.importorskip('mujoco')
from simaudit import check_order, Status
from simaudit.adapters.mujoco import mujoco_xml


def test_smooth_native_and_unknown_features():
    xml='''<mujoco><option integrator="RK4"/><worldbody><body><joint type="hinge" axis="0 1 0"/>
    <geom type="box" size=".2 .1 .05" pos=".3 0 0" contype="0" conaffinity="0"/>
    </body></worldbody></mujoco>'''
    a=mujoco_xml(xml)
    assert check_order(a,[[.1,.5]],[0],interval=.05).status==Status.PASS
    restricted=mujoco_xml(xml.replace('contype="0" conaffinity="0"',''))
    assert check_order(restricted,[[.1,.5]],[0]).status==Status.PARTIAL


def test_quaternion_sign_metric():
    xml='''<mujoco><worldbody><body><joint type="ball"/><geom type="box" size=".2 .1 .05"
    contype="0" conaffinity="0"/></body></worldbody></mujoco>'''
    a=mujoco_xml(xml)
    assert a.distance(np.array([1.,0,0,0,0,0,0]),np.array([-1.,0,0,0,0,0,0]))<1e-12
