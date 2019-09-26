# TCG_2019_project1
 Threes! game env setting

 TODO list:

 board.h : 
 	1. validation rule for placing a tile on board.(may interact with bag)
 	2. slide mechanism and merge rule

 agent.h :
	1. rndenv(evil) placing mechanism ,implement bag
 (to add)

some observation :
	1. placing rule : d->top row , u->down row , l->right column , r->left column
	2. init board : run 3 times bags randomly
	3. heuristic idea : maybe try to modifiy the reward func. in board.h
	   then we simply choose the best reward