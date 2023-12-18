# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/03.config.vehicles.ipynb.

# %% auto 0
__all__ = ['PEDAL_SCALES', 'SPEED_SCALES_MULE', 'SPEED_SCALES_VB', 'TRIANGLE_TEST_CASE_TARGET_VELOCITIES', 'TruckCat', 'Maturity',
           'RE_VIN', 'trucks', 'trucks_all', 'trucks_by_id', 'trucks_by_vin', 'OperationHistory', 'KvaserMixin',
           'TboxMixin', 'Truck', 'TruckInCloud', 'TruckInField']

# %% ../../nbs/03.config.vehicles.ipynb 2
from dataclasses import dataclass, field
from typing import ClassVar, Optional
from zoneinfo import ZoneInfo  # type: ignore
from ordered_set import OrderedSet
from pandas import Timestamp

# %% ../../nbs/03.config.vehicles.ipynb 3
from ..data.time import timezones
from ..data.location import EosLocation, locations, locations_by_abbr
from pprint import pprint

# %% ../../nbs/03.config.vehicles.ipynb 4
PEDAL_SCALES = (
    0,
    0.02,
    0.04,
    0.08,
    0.12,
    0.16,
    0.20,
    0.24,
    0.28,
    0.32,
    0.38,
    0.44,
    0.50,
    0.62,
    0.74,
    0.86,
    1.0,
)

# %% ../../nbs/03.config.vehicles.ipynb 6
SPEED_SCALES_MULE = (
    0,
    7,
    10,
    15,
    20,
    25,
    30,
    35,
    40,
    45,
    50,
    55,
    60,
    65,
    70,
    75,
    80,
    85,
    90,
    95,
    100,
)  # in km/h, 21 elements

# %% ../../nbs/03.config.vehicles.ipynb 8
SPEED_SCALES_VB = (
    0,
    7,
    10,
    20,
    30,
    40,
    50,
    60,
    70,
    80,
    90,
    100,
    110,
    120,
)  # in km/h, 14 elements

# %% ../../nbs/03.config.vehicles.ipynb 10
TRIANGLE_TEST_CASE_TARGET_VELOCITIES = (
    0,
    1.8,
    3.6,
    5.4,
    7.2,
    9,
    10.8,
    12.6,
    14.4,
    16.2,
    14.4,
    12.6,
    10.8,
    9,
    7.2,
    5.4,
    3.6,
    1.8,
    0,
    0,
    0,
)  # triangle test case in km/h

# %% ../../nbs/03.config.vehicles.ipynb 12
TruckCat = OrderedSet(
    [
        "MP73",
        "MP74",
        "MP02",
        "MP20",
        "MP58",
        "MP57",
        "VB4",
        "VB1",
        "VB97",
        "VB7",
        "VB6",
        "M2",
        "HQB",
    ]
)

# %% ../../nbs/03.config.vehicles.ipynb 14
Maturity = OrderedSet(["MULE", "VB", "MP"])

# %% ../../nbs/03.config.vehicles.ipynb 16
RE_VIN = r"^HMZABAAH\wMF\d{6}$"

# %% ../../nbs/03.config.vehicles.ipynb 18
@dataclass
class OperationHistory:
    """History of the vehicle operation

    Attributes:

        site: location of the vehicle
        date_range: date range of the vehicle operation

    """

    site: Optional[EosLocation] = None
    date_range: tuple[Timestamp, Timestamp] = (
        Timestamp(ts_input="2022-12-01T00:00:00", tz="Asia/Shanghai"),
        Timestamp(ts_input="2023-12-31T00:00:00", tz="Asia/Shanghai"),
    )

# %% ../../nbs/03.config.vehicles.ipynb 19
@dataclass
class KvaserMixin:
    """
    Mixin class for Kvaser interface

    Attributes:

            kvaser_observation_number: number of observation in one unit
            kvaser_observation_frequency: frequency of observation
            kvaser_countdown: countdown time before observation
    """

    # optional: can be adjusted by developer
    kvaser_observation_number: ClassVar[
        int
    ] = 30  # Kvaser number of one observation unit: 30 as count number,
    kvaser_observation_frequency: ClassVar[
        int
    ] = 20  # Kvaser observation frequency: 20 Hz, fixed by hardware setting
    kvaser_countdown: ClassVar[
        int
    ] = 3  # Kvaser countdown time: 3 seconds, optional: can be adjusted by developer

# %% ../../nbs/03.config.vehicles.ipynb 20
@dataclass
class TboxMixin:
    """
    Mixin class for Tbox interface

    fixed by hardware setting of remotecan

    Attributes:

            tbox_signal_frequency: frequency of signal
            tbox_gear_frequency: frequency of gear
            tbox_unit_duration: duration of one unit
            tbox_unit_number: number of units
    """

    tbox_signal_frequency: ClassVar[int] = 50  # Hz
    tbox_gear_frequency: ClassVar[int] = 2  # Hz
    tbox_unit_duration: ClassVar[int] = 1  # cloud unit duration in seconds
    tbox_unit_number: ClassVar[int] = 4  # cloud number of units of cloud observation

# %% ../../nbs/03.config.vehicles.ipynb 21
@dataclass
class Truck:
    """
    Truck class

    Attributes:

        vid: vehicle id (short name) of the truck: VB7, M2, MP2, etc.
        vin: Vehicle Identification Number
        plate: License plate number
        maturity: "VB", "MULE", "MP"
        site: current Location of the truck
        operation_history: list of Operation history of the truck
        interface: interface of the truck, "cloud" or "kvaser"
        tbox_id: Tbox id of the truck
        pedal_scale: percentage of pedal opening [0, 100]
        _torque_table_col_num: number of columns/pedal_scales in the torque map
        speed_scale: range of velocity [0, 100] in km/h
        _torque_table_row_num: number of rows/speed_scales in the torque map
        observation_number: number of observation, 3: velocity, throttle, brake
        torque_budget: maximal delta torque to be overlapped on the torque map 250 in Nm
        torque_lower_bound: minimal percentage of delta torque to be overlapped on the torque map: 0.8
        torque_upper_bound: maximal percentage of delta torque to be overlapped on the torque map: 1.0
        torque_bias: bias of delta torque to be overlapped on the torque map: 0.0
        torque_table_row_num_flash: number of rows to be flashed in the torque map: 4
        cat: category of the truck
        _torque_flash_numel: actually flashed number of torque items in the torque map
        _torque_full_numel: number of full torque items in the torque map
        _observation_numel: number of torque items in the torque map
        _observation_length: number of torque items in the torque map
        _observation_sampling_rate: sampling rate/frequency of the truck
        _observation_duration: sampling rate of the truck
    """

    vid: str  # vehicle id (short name) of the truck: VB7, M2, MP2, etc.
    vin: str  # Vehicle Identification Number
    plate: str  # License plate number
    maturity: str  # "VB", "MULE", "MP"
    site: EosLocation  # current Location of the truck
    operation_history: list[OperationHistory] = field(
        default_factory=list
    )  # list of Operation history of the truck
    interface: str = ""  # interface of the truck, "cloud" or "kvaser"
    tbox_id: Optional[str] = None  # Tbox id of the truck
    pedal_scale: tuple = field(
        default_factory=tuple
    )  # percentage of pedal opening [0, 100]
    _torque_table_col_num: Optional[
        int
    ] = None  # number of columns/pedal_scales in the torque map
    speed_scale: tuple = field(
        default_factory=tuple
    )  # range of velocity [0, 100] in km/h
    _torque_table_row_num: Optional[
        int
    ] = None  # number of rows/speed_scales in the torque map
    observation_number: int = 3  # number of observation, 3: velocity, throttle, brake
    torque_budget: int = (
        250  # maximal delta torque to be overlapped on the torque map 250 in Nm
    )
    # optionally use torque_lower_bound, torque_upper_bound, torque_bias
    torque_lower_bound: float = 0.8  # minimal percentage of delta torque to be overlapped on the torque map: 0.8
    torque_upper_bound: float = 1.0  # maximal percentage of delta torque to be overlapped on the torque map: 1.0
    torque_bias: float = (
        0.0  # bias of delta torque to be overlapped on the torque map: 0.0
    )
    torque_table_row_num_flash: int = (
        4  # number of rows to be flashed in the torque map: 4
    )
    cat: OrderedSet = field(default_factory=OrderedSet)  # category of the truck
    _torque_flash_numel: Optional[
        int
    ] = None  # actually flashed number of torque items in the torque map
    _torque_full_numel: Optional[
        int
    ] = None  # number of full torque items in the torque map
    _observation_numel: Optional[
        float
    ] = None  # number of torque items in the torque map
    _observation_length: Optional[
        int
    ] = None  # number of torque items in the torque map
    _observation_sampling_rate: Optional[
        float
    ] = None  # sampling rate/frequency of the truck
    _observation_duration: Optional[float] = None  # sampling rate of the truck

    def __post_init__(self):
        """post init function to set the attributes of the truck"""
        self.pedal_scale = PEDAL_SCALES
        self.speed_scale = SPEED_SCALES_VB
        self.cat = TruckCat  # OrderedSet() is mutable,
        # so that all object share the same cat, and get updated when new truck is added
        self.cat.add(self.vid)
        self.torque_table_row_num = len(self.speed_scale)
        self.torque_table_col_num = len(self.pedal_scale)
        self.torque_full_numel = self.torque_table_row_num * self.torque_table_col_num
        self.torque_flash_numel = (
            self.torque_table_row_num_flash * self.torque_table_col_num  # 4*17 = 68
        )

    @property
    def torque_flash_numel(self):
        return self._torque_flash_numel

    @torque_flash_numel.setter
    def torque_flash_numel(self, value):
        self._torque_flash_numel = value

    @property
    def torque_full_numel(self):
        return self._torque_full_numel

    @torque_full_numel.setter
    def torque_full_numel(self, value):
        self._torque_full_numel = value

    @property
    def observation_numel(self):
        return self._observation_numel

    @observation_numel.setter
    def observation_numel(self, value):
        self._observation_numel = value

    @property
    def observation_length(self):
        return self._observation_length

    @observation_length.setter
    def observation_length(self, value):
        self._observation_length = value

    @property
    def observation_sampling_rate(self):
        return self._observation_sampling_rate

    @observation_sampling_rate.setter
    def observation_sampling_rate(self, value):
        self._observation_sampling_rate = value

    @property
    def observation_duration(self):
        return self._observation_duration

    @observation_duration.setter
    def observation_duration(self, value):
        self._observation_duration = value

    @property
    def torque_table_row_num(self):
        return self._torque_table_row_num

    @torque_table_row_num.setter
    def torque_table_row_num(self, value):
        self._torque_table_row_num = value

    @property
    def torque_table_col_num(self):
        return self._torque_table_col_num

    @torque_table_col_num.setter
    def torque_table_col_num(self, value):
        self._torque_table_col_num = value

# %% ../../nbs/03.config.vehicles.ipynb 22
@dataclass
class TruckInCloud(TboxMixin, Truck):
    """
    Truck in cloud

    Attributes:

        interface: interface of the truck, "cloud"
        observation_length: length of observation
        observation_numel: number of observation
        observation_sampling_rate: sampling rate of observation
        observation_duration: duration of observation
        torque_table_row_num_flash: number of rows to be flashed in the torque map
    """

    def __post_init__(self):
        super().__post_init__()
        self.interface = "cloud"
        self.observation_length = (
            self.tbox_unit_number * self.tbox_unit_duration * self.tbox_signal_frequency
        )  # 4 * 1 * 50 = 200
        self.observation_numel = (
            self.observation_number * self.observation_length  # 3 * 200 = 600
        )
        self.observation_sampling_rate = self.tbox_signal_frequency
        self.observation_duration = (
            self.tbox_unit_duration * self.tbox_unit_number
        )  # 1 * 4 = 4s
        self.torque_table_row_num_flash = 4

# %% ../../nbs/03.config.vehicles.ipynb 23
@dataclass
class TruckInField(KvaserMixin, Truck):
    """
    Truck in field

    Attributes:

        interface: interface of the truck, "kvaser"
        observation_length: length of observation
        observation_numel: number of observation
        observation_sampling_rate: sampling rate of observation
        observation_duration: duration of observation
        torque_table_row_num_flash: number of rows to be flashed in the torque map
    """

    def __post_init__(self):
        super().__post_init__()
        self.interface = "kvaser"
        self.observation_length = self.kvaser_observation_number
        self.observation_numel = (
            self.observation_number * self.observation_length  # 3* 30 = 90
        )
        self.observation_sampling_rate = self.kvaser_observation_frequency
        self.observation_duration = (
            self.kvaser_observation_number / self.kvaser_observation_frequency
        )  # in seconds, default 1.5s
        self.torque_table_row_num_flash = 4

# %% ../../nbs/03.config.vehicles.ipynb 24
trucks = [
    TruckInCloud(
        vid="default",
        vin="",
        plate="",
        operation_history=[OperationHistory()],
        maturity="",
        site=locations_by_abbr["unknown"],
        tbox_id="",  # TBox ID
    ),
    TruckInCloud(
        vid="MP73",
        vin="HMZABAAH4NF003873",
        plate="沪AB82511",
        operation_history=[
            OperationHistory(
                site=locations_by_abbr["unknown"],
                date_range=(
                    Timestamp(ts_input="2023-05-01T00:00:00", tz="Asia/Shanghai"),
                    Timestamp(ts_input="2023-12-31T00:00:00", tz="Asia/Shanghai"),
                ),
            ),
        ],
        maturity="MP",
        site=locations_by_abbr["unknown"],
        tbox_id="73361466",  # TBox ID
    ),
    TruckInCloud(
        vid="MP74",
        vin="HMZABAAH4MF018274",
        plate="苏EDY8203",
        operation_history=[
            OperationHistory(
                site=locations_by_abbr["unknown"],
                date_range=(
                    Timestamp(ts_input="2023-05-01T00:00:00", tz="Asia/Shanghai"),
                    Timestamp(ts_input="2099-12-31T00:00:00", tz="Asia/Shanghai"),
                ),
            ),
        ],
        maturity="MP",
        site=locations_by_abbr["unknown"],
        tbox_id="73453868",  # TBox ID
    ),
    TruckInCloud(
        vid="MP02",
        vin="HMZABAAH1NF004902",
        plate="沪ABD9968",
        operation_history=[
            OperationHistory(
                site=locations_by_abbr["unknown"],
                date_range=(
                    Timestamp(ts_input="2023-05-01T00:00:00", tz="Asia/Shanghai"),
                    Timestamp(ts_input="2099-12-31T00:00:00", tz="Asia/Shanghai"),
                ),
            ),
        ],
        maturity="MP",
        site=locations_by_abbr["unknown"],
        tbox_id="73453941",  # TBox ID
    ),
    TruckInCloud(
        vid="MP20",
        vin="HMZABAAH9NF005120",
        plate="沪AB87177",
        maturity="MP",
        tbox_id="73454077",  # TBox ID
        operation_history=[
            OperationHistory(
                site=locations_by_abbr["unknown"],
                date_range=(
                    Timestamp(ts_input="2023-05-01T00:00:00", tz="Asia/Shanghai"),
                    Timestamp(ts_input="2099-12-31T00:00:00", tz="Asia/Shanghai"),
                ),
            ),
        ],
        site=locations_by_abbr["unknown"],
    ),
    TruckInCloud(
        vid="MP58",
        vin="HMZABAAHXNF005658",
        plate="苏BDT6566",
        maturity="MP",
        operation_history=[
            OperationHistory(
                site=locations_by_abbr["unknown"],
                date_range=(
                    Timestamp(ts_input="2023-05-01T00:00:00", tz="Asia/Shanghai"),
                    Timestamp(ts_input="2099-12-31T00:00:00", tz="Asia/Shanghai"),
                ),
            ),
        ],
        site=locations_by_abbr["unknown"],
    ),
    TruckInCloud(
        vid="MP57",
        vin="HMZABAAH8NF005657",
        plate="苏BDT6608",
        maturity="MP",
        operation_history=[
            OperationHistory(
                site=locations_by_abbr["unknown"],
                date_range=(
                    Timestamp(ts_input="2023-05-01T00:00:00", tz="Asia/Shanghai"),
                    Timestamp(ts_input="2099-12-31T00:00:00", tz="Asia/Shanghai"),
                ),
            ),
        ],
        site=locations_by_abbr["unknown"],
    ),
    TruckInCloud(
        vid="VB4",
        vin="HMZABAAHXMF011054",
        plate="77777777",
        maturity="VB1",
        operation_history=[
            OperationHistory(
                site=locations_by_abbr["unknown"],
                date_range=(
                    Timestamp(ts_input="2023-05-01T00:00:00", tz="Asia/Shanghai"),
                    Timestamp(ts_input="2099-12-31T00:00:00", tz="Asia/Shanghai"),
                ),
            ),
        ],
        site=locations_by_abbr["jy"],
    ),
    TruckInCloud(
        vid="VB1",
        vin="HMZABAAH1MF011055",
        plate="77777777",
        maturity="VB1",
        operation_history=[
            OperationHistory(
                site=locations_by_abbr["unknown"],
                date_range=(
                    Timestamp(ts_input="2023-05-01T00:00:00", tz="Asia/Shanghai"),
                    Timestamp(ts_input="2099-12-31T00:00:00", tz="Asia/Shanghai"),
                ),
            ),
        ],
        site=locations_by_abbr["jy"],
    ),
    TruckInCloud(
        vid="SU_BDC8937",
        vin="HMZABAAH4MF014497",
        plate="SU-BDC8937",
        maturity="VB",
        operation_history=[
            OperationHistory(
                site=locations_by_abbr["unknown"],
                date_range=(
                    Timestamp(ts_input="2023-05-01T00:00:00", tz="Asia/Shanghai"),
                    Timestamp(ts_input="2099-12-31T00:00:00", tz="Asia/Shanghai"),
                ),
            ),
        ],
        site=locations_by_abbr["jy"],
    ),
    TruckInCloud(
        vid="VB7",
        vin="HMZABAAH7MF011058",
        plate="77777777",
        operation_history=[
            OperationHistory(
                site=locations_by_abbr["at"],
                date_range=(
                    Timestamp(ts_input="2023-05-01T00:00:00", tz="Asia/Shanghai"),
                    Timestamp(ts_input="2099-12-31T00:00:00", tz="Asia/Shanghai"),
                ),
            ),
        ],
        maturity="VB",
        site=locations_by_abbr["at"],
    ),
    TruckInCloud(
        vid="VB6",
        vin="HMZABAAH5MF011057",
        plate="66666666",
        maturity="VB",
        operation_history=[
            OperationHistory(
                site=locations_by_abbr["at"],
                date_range=(
                    Timestamp(ts_input="2023-05-01T00:00:00", tz="Asia/Shanghai"),
                    Timestamp(ts_input="2099-12-31T00:00:00", tz="Asia/Shanghai"),
                ),
            ),
        ],
        site=locations_by_abbr["at"],
    ),
    TruckInCloud(
        vid="M2",
        vin="HMZABAAH5MF000000",  # meaning unknown # "987654321654321M4"
        plate="2222222",
        maturity="MULE",
        operation_history=[
            OperationHistory(
                site=locations_by_abbr["at"],
                date_range=(
                    Timestamp(ts_input="2023-05-01T00:00:00", tz="Asia/Shanghai"),
                    Timestamp(ts_input="2099-12-31T00:00:00", tz="Asia/Shanghai"),
                ),
            ),
        ],
        site=locations_by_abbr["at"],
        speed_scale=SPEED_SCALES_MULE,
    ),
    TruckInCloud(
        vid="HQB",
        vin="HMZABAAH5MF999999",  # meaning fictive vin
        plate="00000000",
        maturity="VB",
        operation_history=[
            OperationHistory(
                site=locations_by_abbr["unknown"],
                date_range=(
                    Timestamp(ts_input="2023-05-01T00:00:00", tz="Asia/Shanghai"),
                    Timestamp(ts_input="2099-12-31T00:00:00", tz="Asia/Shanghai"),
                ),
            ),
        ],
        site=locations_by_abbr["hq"],
    ),  # HQ Bench
]  # default trucks in cloud, trucks doesn't contain the trucks in field

# %% ../../nbs/03.config.vehicles.ipynb 26
trucks_all = [
    *trucks,
    TruckInField(
        vid="VB7_FIELD",
        vin="HMZABAAH7MF011058",
        plate="77777777",
        operation_history=[
            OperationHistory(
                site=locations_by_abbr["at"],
                date_range=(
                    Timestamp(ts_input="2023-05-01T00:00:00", tz="Asia/Shanghai"),
                    Timestamp(ts_input="2099-12-31T00:00:00", tz="Asia/Shanghai"),
                ),
            ),
        ],
        maturity="VB",
        site=locations_by_abbr["at"],
    ),  # VB7 in field with Kvaser interface
]  # include all trucks in cloud and in field

# %% ../../nbs/03.config.vehicles.ipynb 28
trucks_by_id = dict(zip([truck.vid for truck in trucks_all], trucks_all))

# %% ../../nbs/03.config.vehicles.ipynb 30
trucks_by_vin = dict(zip([truck.vin for truck in trucks], trucks))
