from genericpath import samefile
from .cpu import pmf
import numpy as np

class TEDAG_FUNCTION:
    def __init__(self, argument_var_names, result_var_name, dt_multiple, func):
        self.args = argument_var_names
        self.result = result_var_name
        self.dt_multiple = dt_multiple
        self.func = func
        
class TEDAG:
    class TEDAG_NODE:
        def __init__(self, name, state, iteration=-1):
            self.name = name
            self.state = state
            self.iteration = iteration
            self.key = name + str(self.iteration)
            
    def __init__(self, time_step, functions, num_agents=100, observables=[], verbose=False):
        # self.dt is largely irrelevant right now but later we might want to know what the simulation time is to multiply the current iteration by dt
        self.dt = time_step
        
        self.current_nodes = {}
        self.dropped_nodes = {}
        self.observables = observables

        self.num_agents = num_agents
        
        self.verbose = verbose
        
        self.DAG = {}
        for f in functions:
            self.DAG[f.result] = f
        
    def addIntervention(self, variables, state, iteration):
        nodes = [self.TEDAG_NODE(variables[v], state[v], iteration) for v in range(len(variables))]
        
        for n in nodes:
            self.current_nodes[n.key] = n

    def getSubState(self, vars, iteration):
        func_arg_node_keys = [a + str(iteration) for a in vars]  
        return np.array([self.current_nodes[a].state for a in func_arg_node_keys])
        
    @classmethod
    def applyFunction(cls, func, in_state, iteration=0):
        func_arg_node_keys = [a + str(iteration-func.dt_multiple) for a in func.args]   

        input_array = np.array([in_state[a] for a in func_arg_node_keys])
        new_node = TEDAG.TEDAG_NODE(func.result, func.func(input_array), iteration)
        return new_node
            
    def findNextFunctionAndApply(self, iteration):
        for result_var_name, func in self.DAG.items():
            node_key = result_var_name + str(iteration)
            
            # if we already calculated and then dropped this node, ignore it
            if node_key in self.dropped_nodes:
                #if self.verbose:
                #    print("Node", node_key, "in dropped nodes. Skipping.")
                continue
            
            # if we already calculated this node, ignore it
            if node_key in self.current_nodes:
                #if self.verbose:
                #    print("Node", node_key, "already in current nodes. Skipping.")
                continue
            
            if not all([a + str(iteration-func.dt_multiple) in self.current_nodes for a in func.args]): # find a node for which all incoming nodes have already been evaluated for this iteration
                #if self.verbose:
                #    print("Node", node_key, "is missing required args.", [a + str(iteration-func.dt_multiple) for a in func.args if a + str(iteration-func.dt_multiple) not in self.current_nodes],"Skipping.")
                continue
            
            if self.verbose:
                print("In findNextFunctionAndApply, found function", result_var_name, "at iteration", iteration)
            
            # If the function is a no-op (say if the input variable is a constant and we're just moving to the next iteration)
            # then just increment the iteration of the node
            if func.func is None:
                # Wait! If the previous version of this is still required, we need to skip for now.
                if any([result_var_name in self.DAG[f].args and self.DAG[f].dt_multiple > 0 for f in self.DAG if f is not result_var_name and f+str(iteration) not in self.current_nodes and f+str(iteration) not in self.dropped_nodes]):
                    continue
                if self.verbose:
                    print("Function for", result_var_name, "is a no-op. Updating the iteration number and associated keys.")
                self.current_nodes[node_key] = self.current_nodes[result_var_name + str(iteration-func.dt_multiple)]
                self.current_nodes[node_key].iteration = iteration
                self.current_nodes[node_key].key = node_key
                self.current_nodes.pop(result_var_name + str(iteration-func.dt_multiple))
                self.dropped_nodes[result_var_name + str(iteration-func.dt_multiple)] = self.current_nodes[node_key]
                return True
            
            # First, if func.args are in different pmfs, they must be independent and we can build a joint distribution
            # This either happens at the beginning of the run when we're first building the populations
            # Or it happens at the point of an intervention
            func_arg_node_keys = [a + str(iteration-func.dt_multiple) for a in func.args]
            func_arg_nodes = [self.current_nodes[k] for k in func_arg_node_keys]
            
            if self.verbose:
                print("Required input variables are:", func_arg_node_keys)
            
            # Apply the function and build a new node
            input_array = np.array([self.current_nodes[a].state for a in func_arg_node_keys])
            check = np.array(func.func(input_array.T))
            new_node = self.TEDAG_NODE(func.result, check, iteration)
            self.current_nodes[new_node.key] = new_node
            
            nodes_to_remove = []
            # Check if any input nodes are no longer required and marginalise them
            for nk in self.current_nodes:
                node = self.current_nodes[nk]
                still_needed = False
                # For each input node, loop through the functions and see if any still need it
                for result_var_name, func in self.DAG.items():
                    if node.name in func.args:
                        if func.result + str(node.iteration + func.dt_multiple) not in self.current_nodes and func.result + str(node.iteration + func.dt_multiple) not in self.dropped_nodes:
                            still_needed = True
                            break
                        
                if not still_needed:
                    if node.name in self.observables and node.iteration == iteration:
                        continue
                        
                    if self.verbose:
                        print("Node", node.key, "no longer required. Deleting...")
                        
                    # empty the state (so save memory) and pop from current_nodes
                    node.state = None
                    nodes_to_remove += [node.key]
                    
            for k in nodes_to_remove:
                self.dropped_nodes[k] = self.current_nodes[k]
                self.current_nodes.pop(k)
                
            if self.verbose:
                print("Current Nodes:", [n for n in self.current_nodes])
                    
            return True
            
        return False