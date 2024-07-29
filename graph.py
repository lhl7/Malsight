import idautils
import idc
import pydot
import queue
import ida_hexrays
import codecs
import json
import heapq
from lex import *

func_to_pseudocode = {}

class Tarjan:
    def __init__(self, vertices,edges):
            self.vertices = vertices
            self.edges = edges
            self.v_num = len(self.vertices)
            self.dfn = [0] * (self.v_num + 5) #编号为 i 的点的DFS序序号
            self.low = [0] * (self.v_num + 5) #编号为 i 的点所在的强连通分量中所有点中的最小dfn值
            self.d2d = [[] for _ in range(self.v_num)] # d2d[i][j] indicate that vertice i connect to vertice j by edge
            self.index = 0 #栈顶下标
            self.stack = [0] * (self.v_num + 5) #栈
            self.in_stk = [False] * self.v_num #编号为 i 的点是否在栈中
            self.timestamp = 0 #时间戳，用来辅助dfn和low
            self.scc_cnt = 0 #当前缩点序号
            self.idd = [0] * (self.v_num + 5) #每个点所属的强连通分量 ID
            self.sz = [0] * (self.v_num + 5) # 每个强连通分量的大小
            self.dot_list = [[] for _ in range(self.v_num + 5)] # 存储每个强连通分量中的顶点列表
            self.vertices_sorted = [] # 拓扑排序后的顶点列表
            self.final_sorted = [] # 最终排序后的顶点列表
            self.st = [False] * (self.v_num + 5) # 标记顶点是否已访问
            self.root = [] # 入度为 0 的顶点列表（根节点）

            self.tar_vertices = [-1] * (self.v_num + 5) # 强连通分量缩点后的顶点列表
            self.tar_d2d = [[] for _ in range(self.v_num + 5)] # 强连通分量缩点后的边列表
            self.outdegree = [0] * (self.v_num + 5) # 每个强连通分量的出度
            self.indegree = [0] * (self.v_num + 5) # 每个强连通分量的入度

            self.dist = [0x3f3f3f3f] * self.v_num # 最短路径距离初始化为无穷大

    #缩点
    def tarjan(self,v):
        self.timestamp += 1
        self.dfn[v] = self.low[v] = self.timestamp
        self.index += 1
        self.stack[self.index] = v
        self.in_stk[v] = True
        for ver in self.d2d[v]:
            if self.dfn[ver] == 0:
                self.tarjan(ver)
                self.low[v] = min(self.low[v],self.low[ver])
            elif self.in_stk[ver]:
                self.low[v] = min(self.low[v],self.dfn[ver])

        if self.dfn[v] == self.low[v]:
            u = -1
            self.scc_cnt += 1
            while v != u:
                u = self.stack[self.index]
                self.index -= 1
                self.in_stk[u] = False
                self.idd[u] = self.scc_cnt
                self.sz[self.scc_cnt] += 1
                self.dot_list[self.scc_cnt].append(u)

    #逆拓扑排序
    def retopsort(self):
        que = queue.Queue()

        for i in range(1,self.scc_cnt + 1):
            if self.outdegree[i] == 0:
                que.put(i)

        while que.qsize() != 0:
            ver = que.get()
            self.vertices_sorted.append(ver)
            for dot in self.tar_d2d[ver]:
                self.outdegree[dot] -= 1
                if self.outdegree[dot] == 0:
                    que.put(dot)

        with open('topsorted.txt', 'w') as f:
            for func in self.vertices_sorted:
                f.write(f"{func}\n")

    #堆优化dijkstra求多源最短路
    def dijkstra(self):
        self.st = [False] * self.v_num
        priority_queue = []
        with open('root.txt','w') as rt:
            for ver in range(0,self.v_num):
                if self.indegree[ver] == 0:
                    rt.write(f'{self.vertices[ver]}\n')
                    self.root.append(ver)
                    self.dist[ver] = 0
                    heapq.heappush(priority_queue, (0, ver))
        while len(priority_queue) != 0:
            _,topVer = heapq.heappop(priority_queue)
            if self.st[topVer]:
                continue
            self.st[topVer] = True
            for ver in self.d2d[topVer]:
                if self.dist[ver] > self.dist[topVer] + 1:
                    self.dist[ver] = self.dist[topVer] + 1
                    heapq.heappush(priority_queue, (self.dist[ver],ver))



    #dfs求环内遍历顺序
    def dfs(self,idx,dot):
        self.st[idx] = True
        for ver in self.d2d[idx]:
            if self.idd[ver] != dot or self.st[ver]:
                continue
            self.dfs(ver,dot)
        self.final_sorted.append(idx)

    #建缩点后的图
    def build_tarGraph(self):
        for edge in self.edges:
            source_func = self.vertices.index(edge[0])
            dest_func = self.vertices.index(edge[1])
            if self.idd[source_func] == self.idd[dest_func]:
                continue
            self.tar_d2d[self.idd[dest_func]].append(self.idd[source_func])
            self.outdegree[self.idd[source_func]] += 1

    #主函数
    def get_tarjan(self):

        for edge in self.edges:
            if edge[0] == edge[1]:
                continue
            source_func = self.vertices.index(edge[0])
            dest_func = self.vertices.index(edge[1])
            self.d2d[source_func].append(dest_func)
            self.indegree[dest_func] += 1

        for ver in range(0,self.v_num):
            if self.dfn[ver] == 0:
                self.tarjan(ver)

        self.build_tarGraph()

        self.retopsort()

        self.dijkstra()
        
        self.st = [False] * self.v_num
        for dot in self.vertices_sorted:
            mindist = 0x3f3f3f3f + 1
            idx = -1
            for ver in self.dot_list[dot]:
                if self.dist[ver] < mindist:
                    mindist = self.dist[ver]
                    idx = ver
            self.dfs(idx,dot)

        with open('final_topsorted.txt', 'w') as f:
            for func in self.final_sorted:
                f.write(f"{self.vertices[func]}\n")

        with open('dist.txt', 'w') as f:
            for it in self.dist:
                f.write(f"{it}\n")

        with open('tarjan.txt','w') as f:
            for i in range(1,self.scc_cnt + 1):
                f.write(f"{i}:\n")
                for j in self.dot_list[i]:
                    f.write(f"{self.vertices[j]} ")
                f.write("\n\n")
#####



def get_function_color(func_name):
    if func_name.startswith('sub_'):
        return 'red'  # 假设以 'sub_' 开头的函数为用户定义函数
    elif func_name.startswith('main'):
        return 'green'  # 假设 'main' 是主函数
    elif func_name.startswith('printf'):
        return 'blue'  # 假设 'printf' 是库函数
    else:
        return 'gray'  # 其他函数

def get_pseu_file(funcname):
    sv = func_to_pseudocode[funcname]
    with open('pesucode.txt','w') as file:
        for sline in sv:
            file.write(f"{ida_lines.tag_remove(sline.line)}\n")

def get_pseu(funcname):
    content = ""
    sv = func_to_pseudocode[funcname]
    if sv == None:
        return content
    for sline in sv:
        content += f"{ida_lines.tag_remove(sline.line)}\n"
    return content

def get_pseu2token(pseucode):
    tokens = tokenizerLexFrompseu(pseucode)
    return tokens 

def func_graph():
    functions = []

    call_relationships = set()

    callee = {}

    for func_ea in idautils.Functions():
        func_name = idc.get_func_name(func_ea)
        func = ida_funcs.get_func(func_ea)

        cfunc = ida_hexrays.decompile(func)
        if cfunc == None:
            sv = None
        else:
            sv = cfunc.get_pseudocode()
        func_to_pseudocode[func_name] = sv
            
        functions.append(func_name)
        
        for callee_ea in idautils.CodeRefsFrom(func_ea, 0):
            callee_name = idc.get_func_name(callee_ea)
            if callee_name:
                call_relationships.add((func_name, callee_name))

        for caller_ea in idautils.CodeRefsTo(func_ea, 0):
            caller_name = idc.get_func_name(caller_ea)
            if caller_name:
                if (caller_name, func_name) in call_relationships:
                    continue
                call_relationships.add((caller_name, func_name))
                if caller_name in callee:
                    callee[caller_name].append(func_name)
                else:
                    callee[caller_name] = []
                    callee[caller_name].append(func_name)

    with open("call.json",'w') as cal:
        for key, value in callee.items():
            fun_callee = {'func_name':key,'callee_list':value}
            json.dump(fun_callee,cal)
            cal.write('\n')

    print("Functions:")
    for func in functions:
        print(func)

    print("\nCall Relationships:")
    for caller, callee in call_relationships:
        print(f"{caller} -> {callee}")

    with open('functions.txt', 'w') as f:
        for func in functions:
            f.write(f"{func}\n")

    with open('call_relationships.txt', 'w') as f:
        for caller, callee in call_relationships:
            f.write(f"{caller} -> {callee}\n")

    tar = Tarjan(functions, call_relationships)
    tar.get_tarjan()

    list_500 = []
    with open('func_tokens.json','w') as ft,open('tokens_cnt.json','w') as tc:
        for func_name in functions:
            pseucode = get_pseu(func_name)
            tokens = get_pseu2token(pseucode)
            fun_token = {'func_name':func_name,'pesucode_tokens':tokens}
            json.dump(fun_token,ft)
            ft.write('\n')
            token_cnt = {'func_name':func_name,'token_cnt':len(tokens)}
            json.dump(token_cnt,tc)
            tc.write('\n')

            if len(tokens) > 500:
                list_500.append(token_cnt)

    with open('token_cnt_500UP.json','w') as tc:
        for sample in list_500:
            json.dump(sample,tc)
            tc.write('\n')

def test():
    functions = ['haha','oo','start','ovo','wow']
    call_relationships = [['start','haha'],['haha','oo'],['oo','ovo'],['ovo','haha'],['ovo','wow'],['wow','oo']]
    tar = Tarjan(functions, call_relationships)
    tar.get_tarjan()
            

    # graph = pydot.Dot(graph_type='digraph')

    # nodes = {}
    # for func in functions:
    #     color = get_function_color(func)
    #     node = pydot.Node(func, style='filled', fillcolor=color)
    #     graph.add_node(node)
    #     nodes[func] = node

    # for caller, callee in call_relationships:
    #     edge = pydot.Edge(nodes[caller], nodes[callee])
    #     graph.add_edge(edge)

    # output_file = 'function_call_graph.svg'
    # graph.write_svg(output_file)
    # print(f"Function call graph saved to {output_file}")


func_graph()
#pseucode = get_pseu('sub_404480')
#tokens = get_pseu2token(pseucode)