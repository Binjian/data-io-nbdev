# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/03.config.drivers.ipynb.

# %% auto 0
__all__ = ['DriverCat', 'RE_DRIVER', 'drivers', 'drivers_by_id', 'Driver']

# %% ../../nbs/03.config.drivers.ipynb 2
from dataclasses import dataclass, field
from ordered_set import OrderedSet
from pandas import Timestamp

# %% ../../nbs/03.config.drivers.ipynb 3
from ..data.location import EosLocation, locations_by_abbr

# %% ../../nbs/03.config.drivers.ipynb 5
DriverCat = OrderedSet(
    [
        "wang-kai",
        "wang-cheng",
        "li-changlong",
        "chen-hongmei",
        "zheng-longfei",
        "UNKNOWN-HUABD9968",
        "UNKNOWN-SUEDY8203",
        "UNKNOWN-HUAB82511",
        "UNKNOWN-HUAB87177",
    ]
)

# %% ../../nbs/03.config.drivers.ipynb 6
RE_DRIVER = r"^[A-Za-z]{1,10}[-_.][A-Za-z]{1,10}(\d?){1,5}$"

# %% ../../nbs/03.config.drivers.ipynb 7
@dataclass
class Driver:
    """
    Driver configuration

    Attributes:

        pid: driver id
        name: driver name
        site: driver location
        contract_range: contract range
        cat: driver category
    """

    pid: str
    name: str
    site: EosLocation
    contract_range: tuple[Timestamp, Timestamp] = (
        Timestamp(ts_input="2022-12-01T00:00:00", tz="Asia/Shanghai"),
        Timestamp(ts_input="2032-12-31T00:00:00+08:00", tz="Asia/Shanghai"),
    )
    cat: OrderedSet = field(default_factory=OrderedSet)

    def __post_init__(self):
        """add DriverCat to cat"""

        self.cat = (
            DriverCat  # OrderedSet is mutable, all objects sharing the same DriverCat
        )
        self.cat.add(self.pid)

# %% ../../nbs/03.config.drivers.ipynb 8
drivers = [
    Driver(
        pid="default",
        name="",
        site=locations_by_abbr["unknown"],
    ),
    Driver(
        pid="wang-kai",
        name="王凯",
        site=locations_by_abbr["at"],
        contract_range=(
            Timestamp(ts_input="2023-08-22T00:00:00+08:00", tz="Asia/Shanghai"),
            Timestamp("2023-09-15T00:00:00+08:00", tz="Asia/Shanghai"),
        ),
    ),
    Driver(
        pid="wang-cheng",
        name="王成",
        site=locations_by_abbr["jy"],
    ),
    Driver(
        pid="li-changlong",
        name="李长龙",
        site=locations_by_abbr["jy"],
    ),
    Driver(
        pid="hongmei-chen",
        name="陈红梅",
        site=locations_by_abbr["jy"],
    ),
    Driver(
        pid="zheng-longfei",
        name="郑龙飞",
        contract_range=(
            Timestamp(ts_input="2022-12-01T00:00:00+08:00", tz="Asia/Shanghai"),
            Timestamp(ts_input="2023-02-15T00:00:00+08:00", tz="Asia/Shanghai"),
        ),
        site=locations_by_abbr["at"],
    ),
    Driver(
        pid="UNKNOWN-HUABD9968",
        name="无名",
        site=locations_by_abbr["unknown"],
    ),
    Driver(
        pid="UNKNOWN-SUEDY8203",
        name="无名",
        site=locations_by_abbr["unknown"],
    ),
    Driver(
        pid="UNKNOWN-HUAB82511",
        name="无名",
        site=locations_by_abbr["unknown"],
    ),
    Driver(
        pid="UNKNOWN-HUAB87177",
        name="无名",
        site=locations_by_abbr["unknown"],
    ),
]

# %% ../../nbs/03.config.drivers.ipynb 9
drivers_by_id = dict(zip([drv.pid for drv in drivers], drivers))