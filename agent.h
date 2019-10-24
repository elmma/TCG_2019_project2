#pragma once
#include <string>
#include <random>
#include <sstream>
#include <map>
#include <type_traits>
#include <algorithm>
#include "board.h"
#include "action.h"
#include "weight.h"
#include <fstream>

class agent {
public:
	agent(const std::string& args = "") {
		std::stringstream ss("name=unknown role=unknown " + args);
		for (std::string pair; ss >> pair; ) {
			std::string key = pair.substr(0, pair.find('='));
			std::string value = pair.substr(pair.find('=') + 1);
			meta[key] = { value };
		}
	}
	virtual ~agent() {}
	virtual void open_episode(const std::string& flag = "") {}
	virtual void close_episode(const std::string& flag = "") {}
	virtual action take_action(const board& b) { return action(); }
	virtual bool check_for_win(const board& b) { return false; }

public:
	virtual std::string property(const std::string& key) const { return meta.at(key); }
	virtual void notify(const std::string& msg) { meta[msg.substr(0, msg.find('='))] = { msg.substr(msg.find('=') + 1) }; }
	virtual std::string name() const { return property("name"); }
	virtual std::string role() const { return property("role"); }

protected:
	typedef std::string key;
	struct value {
		std::string value;
		operator std::string() const { return value; }
		template<typename numeric, typename = typename std::enable_if<std::is_arithmetic<numeric>::value, numeric>::type>
		operator numeric() const { return numeric(std::stod(value)); }
	};
	std::map<key, value> meta;
};

class random_agent : public agent {
public:
	random_agent(const std::string& args = "") : agent(args) {
		if (meta.find("seed") != meta.end())
			engine.seed(int(meta["seed"]));
	}
	virtual ~random_agent() {}

protected:
	std::default_random_engine engine;
};


/**
 * base agent for agents with weight tables
 */
class weight_agent : public random_agent {
public:
	weight_agent(const std::string& args = "") : agent(args) {
		if (meta.find("init") != meta.end()) // pass init=... to initialize the weight
			init_weights(meta["init"]);
		if (meta.find("load") != meta.end()) // pass load=... to load from a specific file
			load_weights(meta["load"]);
	}
	virtual ~weight_agent() {
		if (meta.find("save") != meta.end()) // pass save=... to save to a specific file
			save_weights(meta["save"]);
	}

protected:
	virtual void init_weights(const std::string& info) {
		// add : actually we just need 15^4 for threes instead of 2^16 for 2048
		// here we use 2^16 for converting the index easier
		net.emplace_back(65536); 
		net.emplace_back(65536); 
		net.emplace_back(65536); 
		net.emplace_back(65536); 
		net.emplace_back(65536); 
		net.emplace_back(65536); 
		net.emplace_back(65536); 
		net.emplace_back(65536); 
		// now net.size() == 2; net[0].size() == 65536; net[1].size() == 65536
	}
	virtual void load_weights(const std::string& path) {
		std::ifstream in(path, std::ios::in | std::ios::binary);
		if (!in.is_open()) std::exit(-1);
		uint32_t size;
		in.read(reinterpret_cast<char*>(&size), sizeof(size));
		net.resize(size);
		for (weight& w : net) in >> w;
		in.close();
	}
	virtual void save_weights(const std::string& path) {
		std::ofstream out(path, std::ios::out | std::ios::binary | std::ios::trunc);
		if (!out.is_open()) std::exit(-1);
		uint32_t size = net.size();
		out.write(reinterpret_cast<char*>(&size), sizeof(size));
		for (weight& w : net) out << w;
		out.close();
	}

protected:
	std::vector<weight> net;
};

/**
 * base agent for agents with a learning rate
 * (add) we modify here
 */
class learning_agent : public weight_agent {
public:
	learning_agent(const std::string& args = "") : weight_agent(args), alpha(0.1f) {
		if (meta.find("alpha") != meta.end())
			alpha = float(meta["alpha"]);
	}
	virtual ~learning_agent() {}

	// add utilities
	float decode(const board& state, int t1, int t2, int t3, int t4) const{
		// change board info to net index
		uint32_t space[] = {0, 1, 2, 3, 6, 12, 24, 48, 96, 192, 384, 768, 1536, 3072, 6144};
		int s1, s2, s3, s4; s1=s2=s3=s4=0;
		while(state(t1)!=space[s1] && s1 < 15) ++s1; 
		while(state(t2)!=space[s2] && s2 < 15) ++s2; 
		while(state(t3)!=space[s3] && s3 < 15) ++s3; 
		while(state(t4)!=space[s4] && s4 < 15) ++s4; 
		//std::cout << "----------------------------------------------------" <<std::endl;
		//std::cout << (s1) << " " << (s2) << " " << (s3) << " " <<(s4) <<std::endl;
		//std::cout << (s1<<12) + (s2<<8) + (s3<<4) +(s4) <<std::endl;
		//std::cout << "----------------------------------------------------" <<std::endl;
		return (s1<<12) + (s2<<8) + (s3<<4) +(s4);	// use 16 for conv
	}

	float state_value(const board& s) const{
		float V=0;
		V += net[0][decode(s,0,1,2,3)];
		V += net[1][decode(s,4,5,6,7)];
		V += net[2][decode(s,8,9,10,11)];
		V += net[3][decode(s,12,13,14,15)];
		V += net[4][decode(s,0,4,8,12)];
		V += net[5][decode(s,1,5,9,13)];
		V += net[6][decode(s,2,6,10,14)];
		V += net[7][decode(s,3,7,11,15)];
		//std::cout << "----------------------------------------------------" <<std::endl;
		//std::cout << V <<std::endl;
		//std::cout << "----------------------------------------------------" <<std::endl;
		return V;
	}
	// for after state , we only give the evaluation instead of (state,reward) pair
	float update(const board& s, const board& s_after, float reward, bool end){
		// note that terminal state with target 0 : end TRUE -> term
		float delta = reward + 0.5*((end)? (0) : (state_value(s_after))) - state_value(s);
		float V=0; float rate = alpha/8.0; 
		//std::cout << "----------------------------------------------------" <<std::endl;
		//std::cout << "delta : " << delta <<std::endl;
		//std::cout << "before : " << net[0][decode(s,0,1,2,3)] <<std::endl;
		//std::cout << "step: " << rate*delta <<std::endl;
		(net[0][decode(s,0,1,2,3)] += rate*delta);
		(net[1][decode(s,4,5,6,7)] += rate*delta);
		(net[2][decode(s,8,9,10,11)] += rate*delta);
		(net[3][decode(s,12,13,14,15)] += rate*delta);
		(net[4][decode(s,0,4,8,12)] += rate*delta);
		(net[5][decode(s,1,5,9,13)] += rate*delta);
		(net[6][decode(s,2,6,10,14)] += rate*delta);
		(net[7][decode(s,3,7,11,15)] += rate*delta);
		//std::cout << "after : " << net[0][decode(s,0,1,2,3)] <<std::endl;
		//std::cout << "----------------------------------------------------" <<std::endl;
		return V; 	// return updated state value
	}

protected:
	float alpha;
};

/**
 * random environment
 * add a new random tile to an empty cell
 * (comment add): we should replace by bag mechanism here
 * space : board id spec.  self_role : evil
 * we may need to record the player action
 * 
 * record player action as criterion 
 * note that always add tile to boarder
 * randomly shuffle bag when new bag round
 * to reset the parameter, implement close method
 */
class rndenv : public random_agent {
public:
	rndenv(const std::string& args = "") : random_agent("name=random role=environment " + args),
		space({ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 }),idx(0) ,bag({1,2,3}) ,popup(0, 2) {}

	virtual void close_episode(const std::string& flag = "") {
		// reset the evil para. for next ep
		idx = 0;
	}

	virtual action take_action(const board& after) {
		std::shuffle(space.begin(), space.end(), engine);
		board::op last = after.get_last_act();	// pass last act
		for (int pos : space) {
			if (after(pos) != 0) continue;

			// boarder: (0,1,2,3) ,(0,4,8,12) ,(12,13,14,15) ,(3,7,11,15)
			if (last == 0 && pos < 12) continue; 
			if (last == 2 && pos > 3) continue; 
			if (last == 1 && pos % 4) continue; 
			if (last == 3 && (pos+1) % 4) continue; 
			
			if(idx == 0 ) std::shuffle(bag.begin(), bag.end(), engine);
			board::cell tile = bag[idx++];
			idx %= 3;

			return action::place(pos, tile);
		}
		return action();
	}

private:
	std::array<int, 16> space;	
	int idx;					// add
	std::array<int, 3> bag;		// add
	std::uniform_int_distribution<int> popup;
};

/**
 * dummy player
 * select a legal action randomly
 *
 * (comment add):we should give it a new heuristic
 * we pick action by choosing the max afterstate value(reward)
 */
class player : public random_agent {
public:
	player(const std::string& args = "") : random_agent("name=dummy role=player " + args),
		opcode({ 0, 1, 2, 3 }) {}

	virtual action take_action(const board& before) {
		std::shuffle(opcode.begin(), opcode.end(), engine);
		for (int op : opcode) {
			board::reward reward = board(before).slide(op);
			if (reward != -1) return action::slide(op);	
			// in origin , we just pick up a valid slide randomly
		}
		return action();
	}

private:
	std::array<int, 4> opcode;
};

/**
 * learning player : what we use for player
 *
 */
class learning_player : public learning_agent {
public:
	learning_player(const std::string& args = "") : learning_agent("name=learning role=player " + args),
		opcode({ 0, 1, 2, 3 }), round(0) {}

	// consider all possible action, evaluate after state value with reward
	// select the action with largest evaluation, should be careful the terminaal state   
	// evil state space :
	virtual action take_action(const board& before) {
		int best_op = -1; float best_eval = -9999999.0; 
		board after; board::reward reward;
		std::shuffle(opcode.begin(), opcode.end(), engine);
	 	for (int op : opcode) {
			after = board(before);
			reward = after.slide(op);
			// now we have s = before, s'=after, r = reward
			if (reward == -1) continue;	// not valid action
			float eval = static_cast<float>(reward) + state_value(after);
			if(best_eval <= eval) {best_op = op; best_eval = eval;}
		}

		after = board(before);
		reward = after.slide(best_op);

		// update para.
		if(round) {
			if(best_op == -1) update(s_before,after,-100,true);	// terminal : target = 0 , reward = -1
			else update(s_before,after,static_cast<float>(reward),false);
		}
		// update state in player space
		s_before = after;

		return (best_op != -1) ? action::slide(best_op) : action();
		//return action();
	}

private:
	std::array<int, 4> opcode;
	board s_before;	// player space : last state log
	unsigned round;
};
