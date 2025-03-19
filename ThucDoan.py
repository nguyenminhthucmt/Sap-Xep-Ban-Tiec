import streamlit as st
import random
import numpy as np
from deap import base, creator, tools, algorithms

# Bảng điểm mức độ thân thiết
affinity_scores = {
    "Vợ/chồng/người yêu": 2000,  # Vợ/chồng/người yêu
    "Anh/chị/em ruột": 900,  # Anh/chị/em ruột
    "Cha/mẹ – con cái": 700,  # Cha/mẹ – con cái
    "Anh chị em họ": 500,  # Anh chị em họ
    "Dì/chú/bác – cháu": 300,  # Dì/chú/bác – cháu
    "Bạn bè": 100,  # Bạn bè
    "Không quen biết": 0  # Không quen biết
}

# Hàm tính điểm gần gũi của một sơ đồ bàn
def fitness(individual, guests, affinity_matrix, num_tables, constraints, max_table_size):
    tables = {i: [] for i in range(num_tables)}
    for guest, table in zip(guests, individual):
        if len(tables[table]) < max_table_size:
            tables[table].append(guest)
    
    score = 0
    for table in tables.values():
        for i in range(len(table)):
            for j in range(i + 1, len(table)):
                pair = (table[i], table[j])
                reverse_pair = (table[j], table[i])
                score += affinity_matrix.get(pair, 0) + affinity_matrix.get(reverse_pair, 0)
    
    penalty = 0
    for (guest1, guest2, must_separate) in constraints:
        for table in tables.values():
            if guest1 in table and guest2 in table:
                if must_separate:
                    penalty += 1000
                else:
                    score += 500
    
    return score - penalty,

# Hàm chạy thuật toán di truyền
def run_ga(guests, affinity_matrix, max_table_size, constraints):
    num_tables = (len(guests) + max_table_size - 1) // max_table_size
    
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()
    toolbox.register("attr_table", lambda: random.randint(0, num_tables - 1))
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_table, n=len(guests))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxUniform, indpb=0.2)
    toolbox.register("mutate", tools.mutUniformInt, low=0, up=num_tables - 1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", lambda ind: fitness(ind, guests, affinity_matrix, num_tables, constraints, max_table_size))
    
    pop = toolbox.population(n=100)
    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=200, stats=None, verbose=False)
    best_ind = tools.selBest(pop, k=1)[0]
    
    tables = {i: [] for i in range(num_tables)}
    for guest, table in zip(guests, best_ind):
        if len(tables[table]) < max_table_size:
            tables[table].append(guest)
    
    return tables

# Giao diện Streamlit
st.title("📌 Công ty tổ chức sự kiện Thành Đoàn Minh Thức ")

# Nhập danh sách khách
guest_input = st.text_area("Nhập danh sách khách (cách nhau bằng dấu phẩy):", "")
max_table_size = st.number_input("Số người tối đa mỗi bàn:", min_value=1, value=5)

if guest_input:
    guests = [g.strip() for g in guest_input.split(",") if g.strip()]
    
    # Nhập mức độ thân thiết giữa các khách mời
    st.subheader("🔗 Chọn mức độ thân thiết giữa các khách")
    affinity_matrix = {}
    
    for i in range(len(guests)):
        for j in range(i + 1, len(guests)):
            key = (guests[i], guests[j])
            relationship = st.selectbox(f"{guests[i]} - {guests[j]}", 
                                        options=list(affinity_scores.keys()), 
                                        format_func=lambda x: x.replace("_", " ").title())
            affinity_matrix[key] = affinity_scores.get(relationship, 0)

    # Nhập ràng buộc
    st.subheader("🚫 Nhập ràng buộc")
    constraint_input = st.text_area("Nhập ràng buộc (VD: A,B,true hoặc A,B,false):", "")
    constraints = []
    if constraint_input:
        constraint_lines = constraint_input.split(";")
        for line in constraint_lines:
            parts = line.split(",")
            if len(parts) == 3:
                guest1, guest2, must_separate = parts[0].strip(), parts[1].strip(), parts[2].strip().lower() == "true"
                constraints.append((guest1, guest2, must_separate))

    # Chạy thuật toán và hiển thị kết quả
    if st.button("🚀 Sắp xếp bàn"):
        if guests and max_table_size > 0:
            result = run_ga(guests, affinity_matrix, max_table_size, constraints)
            
            st.subheader("📋 Kết quả sắp xếp:")
            for table, guests in result.items():
                st.write(f"**Bàn {table + 1}:** {', '.join(guests) if guests else 'Không có ai'}")
