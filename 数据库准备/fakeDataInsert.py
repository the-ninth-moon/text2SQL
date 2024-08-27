import pymysql
from faker import Faker
import random

# 创建数据库连接
db = pymysql.connect(
    host="47.98.41.200",
    user="root",  # 替换为你的MySQL用户名
    password="@text2SQL",  # 替换为你的MySQL密码
    database="text2sql",
    charset='utf8mb4'
)

cursor = db.cursor()

# 初始化faker
fake = Faker('zh_CN')

# 插入data_info_by_city表的假数据
for _ in range(100):
    city = fake.city_name()
    area = fake.district()
    province = fake.province()
    cumulative_charging_swapping_times = random.randint(1000, 10000)
    cumulative_charging_times = random.randint(500, 5000)
    cumulative_swapping_times = random.randint(500, 5000)
    daily_real_time_charging_swapping_times = random.randint(50, 500)
    daily_real_time_charging_times = random.randint(25, 250)
    daily_real_time_swapping_times = random.randint(25, 250)
    battery_safety_rate = random.choice([True, False])
    battery_non_safety_rate = not battery_safety_rate
    reduce_total_mileage = f"{random.randint(100, 1000)} km"
    reduce_carbon_emissions = f"{random.randint(10, 100)} tons"
    battery_riding_mileage = random.randint(10, 100)
    battery_non_safety_times = random.randint(1, 10)

    cursor.execute(
        "INSERT INTO data_info_by_city (city, area, province, cumulative_charging_swapping_times, cumulative_charging_times, "
        "cumulative_swapping_times, daily_real_time_charging_swapping_times, daily_real_time_charging_times, "
        "daily_real_time_swapping_times, battery_safety_rate, battery_non_safety_rate, reduce_total_mileage, "
        "reduce_carbon_emissions, battery_riding_mileage, battery_non_safety_times) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
        (city, area, province, cumulative_charging_swapping_times, cumulative_charging_times,
         cumulative_swapping_times, daily_real_time_charging_swapping_times, daily_real_time_charging_times,
         daily_real_time_swapping_times, battery_safety_rate, battery_non_safety_rate, reduce_total_mileage,
         reduce_carbon_emissions, battery_riding_mileage, battery_non_safety_times)
    )

# 插入user_rider_info表的假数据
for _ in range(100):
    city = fake.city_name()
    area = fake.district()
    province = fake.province()
    gender = random.choice([0, 1])
    age = random.randint(18, 60)
    hd_count = random.randint(10, 100)
    daily_hd_count = random.randint(1, 10)

    cursor.execute(
        "INSERT INTO user_rider_info (city, area, province, gender, age, hd_count, daily_hd_count) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s)",
        (city, area, province, gender, age, hd_count, daily_hd_count)
    )

# 插入battery表的假数据
for _ in range(100):
    city = fake.city_name()
    area = fake.district()
    province = fake.province()
    is_qishou = random.choice([True, False])
    is_guizi = not is_qishou
    battery_type = random.randint(1, 4)
    status = random.randint(0, 2)

    cursor.execute(
        "INSERT INTO battery (city, area, province, is_qishou, is_guizi, type, status) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s)",
        (city, area, province, is_qishou, is_guizi, battery_type, status)
    )

# 插入hdg_info表的假数据
for _ in range(100):
    city = fake.city_name()
    area = fake.district()
    province = fake.province()

    cursor.execute(
        "INSERT INTO hdg_info (city, area, province) "
        "VALUES (%s, %s, %s)",
        (city, area, province)
    )

# 提交所有操作
db.commit()

# 关闭连接
cursor.close()
db.close()

print("数据插入完成!")
