from .cpu import pmf

class TEDAG_FUNCTION:
    def __init__(self, argument_var_names, result_var_name, dt_multiple, transition, output_pmf_template):
        self.args = argument_var_names
        self.result = result_var_name
        self.dt_multiple = dt_multiple
        self.transition = transition
        self.output_pmf_template = output_pmf_template
        self.input_pmf = None
        self.output_pmf = None
        
class TEDAG:
    class TEDAG_NODE:
        def __init__(self, name, iteration=-1, pmf_index=None):
            self.name = name
            self.iteration = iteration
            self.pmf_index = pmf_index
            self.key = name + str(self.iteration)
            
    class TEDAG_PMF:
        def __init__(self, pmf, nodes):
            self.pmf = pmf
            self.nodes = nodes
            
        def getDimensionsOfVariables(self, names):
            node_names = [n.name for n in self.nodes]
            dims = []
            for name in names:
                if name in node_names:
                    dims += [node_names.index(name)]
            return dims
                
            
    def __init__(self, time_step, functions, observables=[]):
        # self.dt is largely irrelevant right now but later we might want to know what the simulation time is to multiply the current iteration by dt
        self.dt = time_step
        
        self.pmfs = {}
        self.pmf_counter = 0
        self.interventions = {}
        self.current_nodes = {}
        self.dropped_nodes = {}
        self.observables = observables
        
        self.DAG = {}
        for f in functions:
            self.DAG[f.result] = f
            
    def getPmfForIteration(self, names, iteration):
        return self.getPmf([(n,iteration) for n in names])
    
    def getPmf(self, names_iterations):
        keys = [n+str(i) for (n,i) in names_iterations]
        pmf_indices = set([self.current_nodes[k].pmf_index for k in keys])
        pmfs = [self.pmfs[index] for index in pmf_indices]
        reduced_pmfs = []
        nodes = []
        for p in pmfs:
            node_keys = [n.key for n in p.nodes if n.key in keys]
            reduced_pmf = p.pmf.calcMarginalToPmf([i for i in range(len(p.nodes)) if p.nodes[i].key in node_keys])
            reduced_pmfs += [reduced_pmf]
            nodes += [self.current_nodes[n] for n in node_keys]
            
        if len(reduced_pmfs) > 1:
            return self.TEDAG_PMF(pmf.buildFromIndependentPmfs(reduced_pmfs), nodes)
        elif len(reduced_pmfs) == 1:
            return self.TEDAG_PMF(reduced_pmfs[0], nodes)    
        else:
            return None
    
        
    def addIntervention(self, variables, iteration, pmf):
        nodes = [self.TEDAG_NODE(v, iteration, self.pmf_counter) for v in variables]
        self.pmfs[self.pmf_counter] = self.TEDAG_PMF(pmf, nodes)
        
        # Add nodes to the current nodes list to show that they have been calculated and stored in a pmf
        # but not yet marginalised out.
        for n in nodes:
            n.pmf_index = self.pmf_counter
            self.current_nodes[n.key] = n
        
        if iteration not in self.interventions:
            self.interventions[iteration] = [self.pmf_counter]
        else:
            self.interventions[iteration] += [self.pmf_counter]
            
        self.pmf_counter += 1
            
    def findNextFunctionAndApply(self, iteration):
        for result_var_name, func in self.DAG.items():
            node_key = result_var_name + str(iteration)
            
            # if we already calculated and then dropped this node, ignore it
            if node_key in self.dropped_nodes:
                continue
            
            # if we already calculated this node, ignore it
            if node_key in self.current_nodes:
                continue
            
            if not all([a + str(iteration-func.dt_multiple) in self.current_nodes for a in func.args]): # find a node for which all incoming nodes have already been evaluated for this iteration
                continue
            
            # If the function is a no-op (say if the input variable is a constant and we're just moving to the next iteration)
            # then just increment the iteration of the node
            if func.transition is None:
                self.current_nodes[node_key] = self.current_nodes[result_var_name + str(iteration-func.dt_multiple)]
                self.current_nodes[node_key].iteration = iteration
                self.current_nodes[node_key].key = node_key
                self.current_nodes.pop(result_var_name + str(iteration-func.dt_multiple))
                self.dropped_nodes[result_var_name + str(iteration-func.dt_multiple)] = self.current_nodes[node_key]
                return True
            
            # First, if func.args are in different pmfs, they must be independent and we can build a joint distribution
            # This either happens at the beginning of the run when we're first building the pmfs
            # Or it happens at the point of an intervention - either way, once we're done, set the function
            # input_pmf to this pmf
            func_arg_node_keys = [a + str(iteration-func.dt_multiple) for a in func.args]
            func_arg_nodes = [self.current_nodes[k] for k in func_arg_node_keys]
            
            involved_pmf_inds = []
            for key in func_arg_node_keys:
                if self.current_nodes[key].pmf_index not in involved_pmf_inds:
                    involved_pmf_inds += [self.current_nodes[key].pmf_index]
                    
            input_pmf_index = involved_pmf_inds[0]
            args_pmf = self.pmfs[input_pmf_index].pmf
            rebuilt_input = len(involved_pmf_inds) > 1
            if len(involved_pmf_inds) > 1:
                args_pmf = pmf.buildFromIndependentPmfs([self.pmfs[a].pmf for a in involved_pmf_inds])
                # build new nodes list for new pmf
                pmf_nodes = []
                
                # add this pmf to pmfs and pop the ones we used to build it
                for pmf_ind in involved_pmf_inds:
                    pmf_nodes += self.pmfs[pmf_ind].nodes
                    self.pmfs.pop(pmf_ind)
                    
                # Reassign pmf indices
                for node in pmf_nodes:
                    node.pmf_index = self.pmf_counter
                
                self.pmfs[self.pmf_counter] = self.TEDAG_PMF(args_pmf, pmf_nodes)
                input_pmf_index = self.pmf_counter
                self.pmf_counter += 1
            
            # Set the func pmf input to match the above (note there may be more nodes involved than just what the func requires)
            func.input_pmf = self.TEDAG_PMF(pmf.duplicate(args_pmf), self.pmfs[input_pmf_index].nodes)
            
            in_nodes = func.input_pmf.nodes.copy()
            
            new_node = self.TEDAG_NODE(func.result, iteration)
            self.current_nodes[new_node.key] = new_node
            
            # Build the output pmf and add it to our list
            if rebuilt_input or func.output_pmf is None or len(func.output_pmf.nodes) != len(func.input_pmf.nodes)+1 or not (all([func.input_pmf.nodes[a].key == func.output_pmf.nodes[a].key for a in range(len(func.input_pmf.nodes))]) and func.output_pmf.nodes[-1].key == new_node.key):
                out_pmf = pmf.buildFromIndependentPmfs([args_pmf, func.output_pmf_template])
                out_nodes = in_nodes + [new_node]
                func.output_pmf = self.TEDAG_PMF(out_pmf, out_nodes)
                
            out_nodes = in_nodes + [new_node]
            func.output_pmf.nodes = out_nodes
            
            # Apply the function
            func_input_dims = [[n.key for n in in_nodes].index(arg + str(iteration-func.dt_multiple)) for arg in func.args]

            func.transition.changeInputDimensions(func_input_dims)
            
            # build mapping
            mapping = {}
            in_names = [n.key for n in func.input_pmf.nodes]
            out_names = [n.key for n in func.output_pmf.nodes]

            for i in range(len(in_names)):
                mapping[i] = out_names.index(in_names[i])
                
            func.transition.applyFunction(func.input_pmf.pmf, func.output_pmf.pmf)
            
            # Copy output_pmf to new pmf and dump the input pmf
            self.pmfs[self.pmf_counter] = self.TEDAG_PMF(pmf.duplicate(func.output_pmf.pmf), func.input_pmf.nodes + [new_node])

            for n in self.pmfs[self.pmf_counter].nodes:
                n.pmf_index = self.pmf_counter
                self.current_nodes[n.key].pmf_index = self.pmf_counter
            self.pmf_counter += 1
            self.pmfs.pop(input_pmf_index)
            
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
                        
                    # marginalise from the pmf and pop from current_nodes
                    counter = 0
                    for n in self.pmfs[node.pmf_index].nodes:
                        if n.key == node.key:
                            dim_to_drop = counter
                            break
                        counter += 1
                        
                    reduced_pmf = self.pmfs[node.pmf_index].pmf.calcMarginalToPmf([i for i in range(self.pmfs[node.pmf_index].pmf.dims) if i != dim_to_drop])
                    self.pmfs[node.pmf_index].pmf = reduced_pmf
                    self.pmfs[node.pmf_index].nodes.pop(dim_to_drop)
                    nodes_to_remove += [node.key]
                    
            for k in nodes_to_remove:
                self.dropped_nodes[k] = self.current_nodes[k]
                self.current_nodes.pop(k)
                    
            return True
            
        return False