package models

import (
	"gopkg.in/mgo.v2/bson"
	"sort"
	// "gopkg.in/mgo.v2"
	"log"
	"strings"
	// "math"
	// "time"
	// "reflect"
	// "encoding/json"
	//"strconv"
	"fmt"
	// "mapstructure"
	// "structs"
)

func GrafanaSearchCommander(workweek, category, hardware string)map[string]interface{}{
	session, useBase := judgeDatabase()
	defer session.Close()

	workweek = strings.ToUpper(workweek)
	category = strings.ToLower(category)
	hardware = strings.ToUpper(hardware)

	var processor,database  string
	var refperiodlist []string
	switch hardware{
		case "PVC","ATSM":
			processor = "gpu"
		case "SPR","EMR","ICX","CPX","GNR","SRF":
			processor = "cpu"
		case "DGPU","GPU":
			processor = "gpu"
		default:
			processor = "cpu"
	}
	jsonmap := make(map[string]interface{})
	// model1json := make(map[string]interface{})
	// model0maps := make([]map[string]interface{},0)
	switch category{
		case "llm":
			database = "llm_" + processor + "_v3_verified_transfer"

			vt := session.DB(useBase).C(database)
			
			// st := session.DB(useBase).C("summary_tabledata")
			dataSet := make([]bson.M, 0)
			sureSet := make(map[string]interface{})
			
			// vtTquery := bson.M{"period":period, "hardware":hardware}
			// vtRquery := bson.M{"period":refP, "hardware":hardware}
			// vtTquery := bson.M{"hardware":hardware,"period":bson.M{"$regex": bson.RegEx{Pattern:`/` + workweek + `/`, Options: "i"}}}
			// vtRquery := bson.M{"period":refP,"hardware":hardware}
			
			vtTquery := bson.M{"hardware":hardware,"period":bson.M{"$regex": workweek}}
			vtTerr := vt.Find(vtTquery).All(&dataSet)
			log.Println("workweek",workweek)
			log.Println("vtTerr", vtTerr)
			// log.Println("database", database, workweek, category, hardware, onedate)
			// log.Println("dataset", dataSet)
		
			// summary_table := make(map[string]interface{})
			if vtTerr == nil && len(dataSet) > 0{
				for _, oneJson := range dataSet{
					// log.Println("period",oneJson["period"].(string))
					if strings.Split(oneJson["period"].(string), ".")[1] != "9"{
						refperiodlist = append(refperiodlist, oneJson["period"].(string))
					}

				}

				sort.Strings(refperiodlist)
				selectedperiod := refperiodlist[len(refperiodlist)-1]
				sureQuery := bson.M{"hardware":hardware,"period":selectedperiod}
				surerr := vt.Find(sureQuery).One(&sureSet)
				log.Println("surerr",surerr, selectedperiod)

				jsonmap["week"] = workweek
				// catekey := hardware + "LLM"

				inferset := make(map[string]interface{})
				trainset := make(map[string]interface{})

				infercon := make(map[string]interface{})
				traincon := make(map[string]interface{})

				for _, onetype := range sureSet["general"].([]interface{}){
					if strings.ToLower(strings.Split(onetype.(map[string]interface{})["r2_type"].(string),"@")[0]) == "realtime"{
						inferset = onetype.(map[string]interface{})
					}else if strings.ToLower(strings.Split(onetype.(map[string]interface{})["r2_type"].(string),"@")[0]) == "training"{
						trainset = onetype.(map[string]interface{})
						log.Println("trainset",trainset["r2_type"].(string))
					}
				}

				littlemodelmap := make(map[string]interface{})
				model0map := make(map[string]interface{})
				model1map := make(map[string]interface{})
				
				model2map:= make(map[string]interface{})
				model3map:= make(map[string]interface{})
			
				model0json := make(map[string]interface{})
				// model1json := make(map[string]interface{})
			
				greedymodel0 := make(map[string][]map[string]interface{})
				beammodel0 := make(map[string][]map[string]interface{})
			
				greedymodel1 := make(map[string][]map[string]interface{})
				beammodel1 := make(map[string][]map[string]interface{})
			
				configmap := make(map[string]interface{})

				// for realtime little model greedy/beam   big model greedy/beam
				modellist0 := make([]string, 0) // little model greedy
				modellist1 := make([]string, 0) // little model beam

				modellist2 := make([]string, 0) // big model greedy
				modellist3 := make([]string, 0) // big model beam

				// for training
				// modellist4 := make([]string, 0)

				// for realtime part 
				for _, ival := range inferset["data"].([]interface{}){
					modelsize := ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["model_size"].(string)
					sizenum := StringTransferToFloat64(strings.Split(strings.ToUpper(modelsize), "B")[0])
					if sizenum <= 13{
						// tokenlist := make([]string, 0)
						// // typelist := make([]string, 0)
				
						if ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["beam_search"].(string) == "1" && ival.(map[string]interface{})["batch_size"].(string) == "1" {
							// log.Println("latency 1",ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["latency"].(string))
							// log.Println("case_name 1", ival.(map[string]interface{})["case_name"].(string) + "-" + modelsize)
							// log.Println("modellist0 1", modellist0)
							if checkDuplicateString(modellist0, ival.(map[string]interface{})["case_name"].(string) + "-" + modelsize){
								modellist0 = append(modellist0, ival.(map[string]interface{})["case_name"].(string) + "-" + modelsize)
								typelist0 := make([]string, 0)
								tokenlist0 := make([]string, 0)
								modeltypetoken0 := make(map[string][]string)
								if checkDuplicateString(typelist0, ival.(map[string]interface{})["precision"].(string)){
									typelist0 = append(typelist0, ival.(map[string]interface{})["precision"].(string))
									token := ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["input_tokens"].(string) + "/" + ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["max_new_tokens"].(string)
									// log.Println("token",token)
									maplist := make([]map[string]interface{},0)
									if checkDuplicateString(tokenlist0, token){
										tokenlist0 = append(tokenlist0, token)
										
										detailmap := make(map[string]interface{})
										detailmap["in_out_token"] = token
										detailmap["precision"] = ival.(map[string]interface{})["precision"].(string)
										detailmap["first_1T"] = "/"
										detailmap["next_avg_1T"] = "/"
										detailmap["next_p90_1T"] = "/"
										detailmap["total_1T"] = "/"
										detailmap["first_1C"] = "/"
										detailmap["next_avg_1C"] = "/"
										detailmap["next_p90_1C"] = "/"
										detailmap["total_1C"] = "/"									
										detailmap["target_first"] = "/"  
										detailmap["target_avg_next"] = "/"
										detailmap["target_total"] = "/"	
										detailmap["target_p90_next"] = "/"
										// log.Println("寻找1",ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["latency"].(string))
										if ival.(map[string]interface{})["tile_num"].(string) == "1" && ival.(map[string]interface{})["card"].(string) == "1"{
											detailmap["total_1T"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["latency"].(string)
											detailmap["first_1T"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["first_latency"].(string) 
											detailmap["next_avg_1T"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["avg_latency"].(string) 
											detailmap["next_p90_1T"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["p90_latency"].(string) 

										}else if ival.(map[string]interface{})["tile_num"].(string) != "1" && ival.(map[string]interface{})["card"].(string) == "1"{
											detailmap["total_1C"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["latency"].(string)
											detailmap["first_1C"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["first_latency"].(string) 
											detailmap["next_avg_1C"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["avg_latency"].(string) 
											detailmap["next_p90_1C"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["p90_latency"].(string) 
										}

										detailmap["target_total"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["target_latency"].(string)
										detailmap["target_avg_next"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["target_avg_latency"].(string)
										detailmap["target_first"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["target_first_latency"].(string)

										maplist = append(maplist, detailmap)
										

									}

									
									modeltypetoken0[ival.(map[string]interface{})["precision"].(string)] = tokenlist0
									greedymodel0[ival.(map[string]interface{})["case_name"].(string) + "-" + modelsize] = maplist

								}else{
									tokenlist0 = modeltypetoken0[ival.(map[string]interface{})["precision"].(string)]
									maplist := greedymodel0[ival.(map[string]interface{})["case_name"].(string) + "-" + modelsize]
									token := ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["input_tokens"].(string) + "/" + ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["max_new_tokens"].(string)
									if checkDuplicateString(tokenlist0, token){
										tokenlist0 = append(tokenlist0, token)

										detailmap := make(map[string]interface{})
										detailmap["in_out_token"] = token
										detailmap["precision"] = ival.(map[string]interface{})["precision"].(string)
										detailmap["first_1T"] = "/"
										detailmap["next_avg_1T"] = "/"
										detailmap["next_p90_1T"] = "/"
										detailmap["total_1T"] = "/"
										detailmap["first_1C"] = "/"
										detailmap["next_avg_1C"] = "/"
										detailmap["next_p90_1C"] = "/"
										detailmap["total_1C"] = "/"									
										detailmap["target_first"] = "/"   
										detailmap["target_avg_next"] = "/"
										detailmap["target_total"] = "/"	
										detailmap["target_p90_next"] = "/"
										// log.Println("寻找4",ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["latency"].(string))
										if ival.(map[string]interface{})["tile_num"].(string) == "1" && ival.(map[string]interface{})["card"].(string) == "1"{
											detailmap["total_1T"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["latency"].(string)
											detailmap["first_1T"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["first_latency"].(string) 
											detailmap["next_avg_1T"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["avg_latency"].(string) 
											detailmap["next_p90_1T"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["p90_latency"].(string) 

										}else if ival.(map[string]interface{})["tile_num"].(string) != "1" && ival.(map[string]interface{})["card"].(string) == "1"{
											detailmap["total_1C"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["latency"].(string)
											detailmap["first_1C"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["first_latency"].(string) 
											detailmap["next_avg_1C"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["avg_latency"].(string) 
											detailmap["next_p90_1C"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["p90_latency"].(string) 
										}
										detailmap["target_total"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["target_latency"].(string)
										detailmap["target_avg_next"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["target_avg_latency"].(string)
										detailmap["target_first"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["target_first_latency"].(string)

										maplist = append(maplist, detailmap)

									}else{
										// log.Println("寻找3",ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["latency"].(string))
										maplist := greedymodel0[ival.(map[string]interface{})["case_name"].(string) + "-" + modelsize]
										for _, oneiter := range maplist{
											if oneiter["precision"] == ival.(map[string]interface{})["precision"].(string) && oneiter["in_out_token"] == token{
												if ival.(map[string]interface{})["tile_num"].(string) == "1" && ival.(map[string]interface{})["card"].(string) == "1"{
													oneiter["total_1T"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["latency"].(string)
													oneiter["first_1T"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["first_latency"].(string) 
													oneiter["next_avg_1T"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["avg_latency"].(string) 
													oneiter["next_p90_1T"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["p90_latency"].(string) 
			
												}else if ival.(map[string]interface{})["tile_num"].(string) != "1" && ival.(map[string]interface{})["card"].(string) == "1"{
													oneiter["total_1C"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["latency"].(string)
													oneiter["first_1C"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["first_latency"].(string) 
													oneiter["next_avg_1C"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["avg_latency"].(string) 
													oneiter["next_p90_1C"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["p90_latency"].(string) 
												}
											}
										}


									}
									modeltypetoken0[ival.(map[string]interface{})["precision"].(string)] = tokenlist0
									greedymodel0[ival.(map[string]interface{})["case_name"].(string) + "-" + modelsize] = maplist
								}
								model0map[ival.(map[string]interface{})["case_name"].(string) + "-" + modelsize] = modeltypetoken0
							}else{
								// log.Println("latency 2",ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["latency"].(string))
								
								// log.Println("greedymodel0 2", greedymodel0)
								// log.Println("modeltypetoken0 2", model0map)
								maplist := greedymodel0[ival.(map[string]interface{})["case_name"].(string) + "-" + modelsize]
								modeltypetoken0 := model0map[ival.(map[string]interface{})["case_name"].(string) + "-" + modelsize].(map[string][]string)
								// log.Println("precision 2",modeltypetoken0[ival.(map[string]interface{})["precision"].(string)])
								// log.Println("precision 22", ival.(map[string]interface{})["precision"].(string))
								if modeltypetoken0[ival.(map[string]interface{})["precision"].(string)] != nil{
									// log.Println("why")
								}
								if modeltypetoken0["int8"] == nil{
									// log.Println("why not")
								}

								// if checkDuplicateString(modeltypetoken0[ival.(map[string]interface{})["precision"].(string)], ival.(map[string]interface{})["precision"].(string)){
								if modeltypetoken0[ival.(map[string]interface{})["precision"].(string)] == nil{
									token := ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["input_tokens"].(string) + "/" + ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["max_new_tokens"].(string)
									tokenlist0 := modeltypetoken0[ival.(map[string]interface{})["precision"].(string)]
									// log.Println("tokenlist0 3",tokenlist0)
									if checkDuplicateString(tokenlist0, token){
										tokenlist0 = append(tokenlist0, token)
										detailmap := make(map[string]interface{})
										detailmap["in_out_token"] = token
										detailmap["precision"] = ival.(map[string]interface{})["precision"].(string)
										detailmap["first_1T"] = "/"
										detailmap["next_avg_1T"] = "/"
										detailmap["next_p90_1T"] = "/"
										detailmap["total_1T"] = "/"
										detailmap["first_1C"] = "/"
										detailmap["next_avg_1C"] = "/"
										detailmap["next_p90_1C"] = "/"
										detailmap["total_1C"] = "/"									
										detailmap["target_first"] = "/"  
										detailmap["target_avg_next"] = "/"
										detailmap["target_total"] = "/"	
										detailmap["target_p90_next"] = "/"
										// log.Println("寻找2",ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["latency"].(string))
										if ival.(map[string]interface{})["tile_num"].(string) == "1" && ival.(map[string]interface{})["card"].(string) == "1"{
											detailmap["total_1T"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["latency"].(string)
											detailmap["first_1T"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["first_latency"].(string) 
											detailmap["next_avg_1T"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["avg_latency"].(string) 
											detailmap["next_p90_1T"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["p90_latency"].(string) 

										}else if ival.(map[string]interface{})["tile_num"].(string) != "1" && ival.(map[string]interface{})["card"].(string) == "1"{
											detailmap["total_1C"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["latency"].(string)
											detailmap["first_1C"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["first_latency"].(string) 
											detailmap["next_avg_1C"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["avg_latency"].(string) 
											detailmap["next_p90_1C"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["p90_latency"].(string) 
										}

										detailmap["target_total"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["target_latency"].(string)
										detailmap["target_avg_next"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["target_avg_latency"].(string)
										detailmap["target_first"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["target_first_latency"].(string)

										maplist = append(maplist, detailmap)


									}
									modeltypetoken0[ival.(map[string]interface{})["precision"].(string)] = tokenlist0
									greedymodel0[ival.(map[string]interface{})["case_name"].(string) + "-" + modelsize] = maplist

								}else{
									maplist := greedymodel0[ival.(map[string]interface{})["case_name"].(string) + "-" + modelsize]

									tokenlist0 := modeltypetoken0[ival.(map[string]interface{})["precision"].(string)]
									token := ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["input_tokens"].(string) + "/" + ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["max_new_tokens"].(string)
									// log.Println("tokenlist0",tokenlist0)
									if checkDuplicateString(tokenlist0, token){
										tokenlist0 = append(tokenlist0, token)
										
										detailmap := make(map[string]interface{})
										detailmap["in_out_token"] = token
										detailmap["precision"] = ival.(map[string]interface{})["precision"].(string)
										detailmap["first_1T"] = "/"
										detailmap["next_avg_1T"] = "/"
										detailmap["next_p90_1T"] = "/"
										detailmap["total_1T"] = "/"
										detailmap["first_1C"] = "/"
										detailmap["next_avg_1C"] = "/"
										detailmap["next_p90_1C"] = "/"
										detailmap["total_1C"] = "/"									
										detailmap["target_first"] = "/"   
										detailmap["target_avg_next"] = "/"
										detailmap["target_total"] = "/"	
										detailmap["target_p90_next"] = "/"
										// log.Println("寻找5",ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["latency"].(string))
										if ival.(map[string]interface{})["tile_num"].(string) == "1" && ival.(map[string]interface{})["card"].(string) == "1"{
											detailmap["total_1T"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["latency"].(string)
											detailmap["first_1T"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["first_latency"].(string) 
											detailmap["next_avg_1T"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["avg_latency"].(string) 
											detailmap["next_p90_1T"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["p90_latency"].(string) 

										}else if ival.(map[string]interface{})["tile_num"].(string) != "1" && ival.(map[string]interface{})["card"].(string) == "1"{
											detailmap["total_1C"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["latency"].(string)
											detailmap["first_1C"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["first_latency"].(string) 
											detailmap["next_avg_1C"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["avg_latency"].(string) 
											detailmap["next_p90_1C"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["p90_latency"].(string) 
										}

										detailmap["target_total"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["target_latency"].(string)
										detailmap["target_avg_next"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["target_avg_latency"].(string)
										detailmap["target_first"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["target_first_latency"].(string)

										maplist = append(maplist, detailmap)
										

									}else{
										// log.Println("寻找6",ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["latency"].(string))
										for _, oneiter := range maplist{
											if oneiter["precision"] == ival.(map[string]interface{})["precision"].(string) && oneiter["in_out_token"] == token{
												if ival.(map[string]interface{})["tile_num"].(string) == "1" && ival.(map[string]interface{})["card"].(string) == "1"{
													oneiter["total_1T"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["latency"].(string)
													oneiter["first_1T"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["first_latency"].(string) 
													oneiter["next_avg_1T"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["avg_latency"].(string) 
													oneiter["next_p90_1T"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["p90_latency"].(string) 
			
												}else if ival.(map[string]interface{})["tile_num"].(string) != "1" && ival.(map[string]interface{})["card"].(string) == "1"{
													oneiter["total_1C"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["latency"].(string)
													oneiter["first_1C"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["first_latency"].(string) 
													oneiter["next_avg_1C"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["avg_latency"].(string) 
													oneiter["next_p90_1C"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["p90_latency"].(string) 
												}
											}
										}

									}
									modeltypetoken0[ival.(map[string]interface{})["precision"].(string)] = tokenlist0
									greedymodel0[ival.(map[string]interface{})["case_name"].(string) + "-" + modelsize] = maplist
								}
								model0map[ival.(map[string]interface{})["case_name"].(string) + "-" + modelsize] = modeltypetoken0
							}

						}else if ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["beam_search"].(string) == "4" && ival.(map[string]interface{})["batch_size"].(string) == "1" {
							// beam = 4
							if checkDuplicateString(modellist1, ival.(map[string]interface{})["case_name"].(string) + "-" + modelsize){
								modellist1 = append(modellist1, ival.(map[string]interface{})["case_name"].(string) + "-" + modelsize)
								typelist1 := make([]string, 0)
								tokenlist1 := make([]string, 0)
								modeltypetoken1 := make(map[string][]string)
								if checkDuplicateString(typelist1, ival.(map[string]interface{})["precision"].(string)){
									typelist1 = append(typelist1, ival.(map[string]interface{})["precision"].(string))
									token := ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["input_tokens"].(string) + "/" + ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["max_new_tokens"].(string)
									maplist := make([]map[string]interface{},0)
									// log.Println("token",token)
									if checkDuplicateString(tokenlist1, token){
										tokenlist1 = append(tokenlist1, token)

										detailmap := make(map[string]interface{})
										detailmap["in_out_token"] = token
										detailmap["precision"] = ival.(map[string]interface{})["precision"].(string)
										detailmap["first_1T"] = "/"
										detailmap["next_avg_1T"] = "/"
										detailmap["next_p90_1T"] = "/"
										detailmap["total_1T"] = "/"
										detailmap["first_1C"] = "/"
										detailmap["next_avg_1C"] = "/"
										detailmap["next_p90_1C"] = "/"
										detailmap["total_1C"] = "/"									
										detailmap["target_first"] = "/"   //等会回来补一下字段，新加的
										detailmap["target_avg_next"] = "/"
										detailmap["target_total"] = "/"	
										detailmap["target_p90_next"] = "/"
										// log.Println("寻找1",ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["latency"].(string))
										if ival.(map[string]interface{})["tile_num"].(string) == "1" && ival.(map[string]interface{})["card"].(string) == "1"{
											detailmap["total_1T"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["latency"].(string)
											detailmap["first_1T"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["first_latency"].(string) 
											detailmap["next_avg_1T"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["avg_latency"].(string) 
											detailmap["next_p90_1T"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["p90_latency"].(string) 

										}else if ival.(map[string]interface{})["tile_num"].(string) != "1" && ival.(map[string]interface{})["card"].(string) == "1"{
											detailmap["total_1C"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["latency"].(string)
											detailmap["first_1C"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["first_latency"].(string) 
											detailmap["next_avg_1C"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["avg_latency"].(string) 
											detailmap["next_p90_1C"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["p90_latency"].(string) 
										}

										detailmap["target_total"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["target_latency"].(string)
										detailmap["target_avg_next"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["target_avg_latency"].(string)
										detailmap["target_first"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["target_first_latency"].(string)

										maplist = append(maplist, detailmap)

									}
									modeltypetoken1[ival.(map[string]interface{})["precision"].(string)] = tokenlist1
									beammodel0[ival.(map[string]interface{})["case_name"].(string) + "-" + modelsize] = maplist

								}else{
									tokenlist1 = modeltypetoken1[ival.(map[string]interface{})["precision"].(string)]
									maplist := beammodel0[ival.(map[string]interface{})["case_name"].(string) + "-" + modelsize]
									token := ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["input_tokens"].(string) + "/" + ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["max_new_tokens"].(string)
									if checkDuplicateString(tokenlist1, token){
										tokenlist1 = append(tokenlist1, token)

										detailmap := make(map[string]interface{})
										detailmap["in_out_token"] = token
										detailmap["precision"] = ival.(map[string]interface{})["precision"].(string)
										detailmap["first_1T"] = "/"
										detailmap["next_avg_1T"] = "/"
										detailmap["next_p90_1T"] = "/"
										detailmap["total_1T"] = "/"
										detailmap["first_1C"] = "/"
										detailmap["next_avg_1C"] = "/"
										detailmap["next_p90_1C"] = "/"
										detailmap["total_1C"] = "/"									
										detailmap["target_first"] = "/"   
										detailmap["target_avg_next"] = "/"
										detailmap["target_total"] = "/"	
										detailmap["target_p90_next"] = "/"
										// log.Println("寻找4",ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["latency"].(string))
										if ival.(map[string]interface{})["tile_num"].(string) == "1" && ival.(map[string]interface{})["card"].(string) == "1"{
											detailmap["total_1T"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["latency"].(string)
											detailmap["first_1T"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["first_latency"].(string) 
											detailmap["next_avg_1T"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["avg_latency"].(string) 
											detailmap["next_p90_1T"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["p90_latency"].(string) 

										}else if ival.(map[string]interface{})["tile_num"].(string) != "1" && ival.(map[string]interface{})["card"].(string) == "1"{
											detailmap["total_1C"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["latency"].(string)
											detailmap["first_1C"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["first_latency"].(string) 
											detailmap["next_avg_1C"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["avg_latency"].(string) 
											detailmap["next_p90_1C"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["p90_latency"].(string) 
										}
										detailmap["target_total"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["target_latency"].(string)
										detailmap["target_avg_next"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["target_avg_latency"].(string)
										detailmap["target_first"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["target_first_latency"].(string)

										maplist = append(maplist, detailmap)

									}else{
										maplist := beammodel0[ival.(map[string]interface{})["case_name"].(string) + "-" + modelsize]
										for _, oneiter := range maplist{
											if oneiter["precision"] == ival.(map[string]interface{})["precision"].(string) && oneiter["in_out_token"] == token{
												if ival.(map[string]interface{})["tile_num"].(string) == "1" && ival.(map[string]interface{})["card"].(string) == "1"{
													oneiter["total_1T"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["latency"].(string)
													oneiter["first_1T"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["first_latency"].(string) 
													oneiter["next_avg_1T"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["avg_latency"].(string) 
													oneiter["next_p90_1T"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["p90_latency"].(string) 
			
												}else if ival.(map[string]interface{})["tile_num"].(string) != "1" && ival.(map[string]interface{})["card"].(string) == "1"{
													oneiter["total_1C"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["latency"].(string)
													oneiter["first_1C"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["first_latency"].(string) 
													oneiter["next_avg_1C"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["avg_latency"].(string) 
													oneiter["next_p90_1C"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["p90_latency"].(string) 
												}
											}
										}

									}
									modeltypetoken1[ival.(map[string]interface{})["precision"].(string)] = tokenlist1
									beammodel0[ival.(map[string]interface{})["case_name"].(string) + "-" + modelsize] = maplist
								}
								model1map[ival.(map[string]interface{})["case_name"].(string) + "-" + modelsize] = modeltypetoken1
							}else{

								maplist := beammodel0[ival.(map[string]interface{})["case_name"].(string) + "-" + modelsize]
								modeltypetoken1 := model1map[ival.(map[string]interface{})["case_name"].(string) + "-" + modelsize].(map[string][]string)

								if modeltypetoken1[ival.(map[string]interface{})["precision"].(string)] == nil{
									token := ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["input_tokens"].(string) + "/" + ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["max_new_tokens"].(string)
									tokenlist1 := modeltypetoken1[ival.(map[string]interface{})["precision"].(string)]
									if checkDuplicateString(tokenlist1, token){
										tokenlist1 = append(tokenlist1, token)

										detailmap := make(map[string]interface{})
										detailmap["in_out_token"] = token
										detailmap["precision"] = ival.(map[string]interface{})["precision"].(string)
										detailmap["first_1T"] = "/"
										detailmap["next_avg_1T"] = "/"
										detailmap["next_p90_1T"] = "/"
										detailmap["total_1T"] = "/"
										detailmap["first_1C"] = "/"
										detailmap["next_avg_1C"] = "/"
										detailmap["next_p90_1C"] = "/"
										detailmap["total_1C"] = "/"									
										detailmap["target_first"] = "/"   //等会回来补一下字段，新加的
										detailmap["target_avg_next"] = "/"
										detailmap["target_total"] = "/"	
										detailmap["target_p90_next"] = "/"
										// log.Println("寻找2",ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["latency"].(string))
										if ival.(map[string]interface{})["tile_num"].(string) == "1" && ival.(map[string]interface{})["card"].(string) == "1"{
											detailmap["total_1T"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["latency"].(string)
											detailmap["first_1T"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["first_latency"].(string) 
											detailmap["next_avg_1T"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["avg_latency"].(string) 
											detailmap["next_p90_1T"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["p90_latency"].(string) 

										}else if ival.(map[string]interface{})["tile_num"].(string) != "1" && ival.(map[string]interface{})["card"].(string) == "1"{
											detailmap["total_1C"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["latency"].(string)
											detailmap["first_1C"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["first_latency"].(string) 
											detailmap["next_avg_1C"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["avg_latency"].(string) 
											detailmap["next_p90_1C"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["p90_latency"].(string) 
										}

										detailmap["target_total"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["target_latency"].(string)
										detailmap["target_avg_next"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["target_avg_latency"].(string)
										detailmap["target_first"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["target_first_latency"].(string)

										maplist = append(maplist, detailmap)

									}
									modeltypetoken1[ival.(map[string]interface{})["precision"].(string)] = tokenlist1
									beammodel0[ival.(map[string]interface{})["case_name"].(string) + "-" + modelsize] = maplist

								}else{
									maplist := beammodel0[ival.(map[string]interface{})["case_name"].(string) + "-" + modelsize]
									tokenlist1 := modeltypetoken1[ival.(map[string]interface{})["precision"].(string)]
									token := ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["input_tokens"].(string) + "/" + ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["max_new_tokens"].(string)
									if checkDuplicateString(tokenlist1, token){
										tokenlist1 = append(tokenlist1, token)

										detailmap := make(map[string]interface{})
										detailmap["in_out_token"] = token
										detailmap["precision"] = ival.(map[string]interface{})["precision"].(string)
										detailmap["first_1T"] = "/"
										detailmap["next_avg_1T"] = "/"
										detailmap["next_p90_1T"] = "/"
										detailmap["total_1T"] = "/"
										detailmap["first_1C"] = "/"
										detailmap["next_avg_1C"] = "/"
										detailmap["next_p90_1C"] = "/"
										detailmap["total_1C"] = "/"									
										detailmap["target_first"] = "/"   //等会回来补一下字段，新加的
										detailmap["target_avg_next"] = "/"
										detailmap["target_total"] = "/"	
										detailmap["target_p90_next"] = "/"
										// log.Println("寻找5",ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["latency"].(string))
										if ival.(map[string]interface{})["tile_num"].(string) == "1" && ival.(map[string]interface{})["card"].(string) == "1"{
											detailmap["total_1T"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["latency"].(string)
											detailmap["first_1T"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["first_latency"].(string) 
											detailmap["next_avg_1T"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["avg_latency"].(string) 
											detailmap["next_p90_1T"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["p90_latency"].(string) 

										}else if ival.(map[string]interface{})["tile_num"].(string) != "1" && ival.(map[string]interface{})["card"].(string) == "1"{
											detailmap["total_1C"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["latency"].(string)
											detailmap["first_1C"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["first_latency"].(string) 
											detailmap["next_avg_1C"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["avg_latency"].(string) 
											detailmap["next_p90_1C"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["p90_latency"].(string) 
										}

										detailmap["target_total"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["target_latency"].(string)
										detailmap["target_avg_next"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["target_avg_latency"].(string)
										detailmap["target_first"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["target_first_latency"].(string)

										maplist = append(maplist, detailmap)

									}else{
										for _, oneiter := range maplist{
											if oneiter["precision"] == ival.(map[string]interface{})["precision"].(string) && oneiter["in_out_token"] == token{
												if ival.(map[string]interface{})["tile_num"].(string) == "1" && ival.(map[string]interface{})["card"].(string) == "1"{
													oneiter["total_1T"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["latency"].(string)
													oneiter["first_1T"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["first_latency"].(string) 
													oneiter["next_avg_1T"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["avg_latency"].(string) 
													oneiter["next_p90_1T"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["p90_latency"].(string) 
			
												}else if ival.(map[string]interface{})["tile_num"].(string) != "1" && ival.(map[string]interface{})["card"].(string) == "1"{
													oneiter["total_1C"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["latency"].(string)
													oneiter["first_1C"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["first_latency"].(string) 
													oneiter["next_avg_1C"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["avg_latency"].(string) 
													oneiter["next_p90_1C"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["p90_latency"].(string) 
												}
											}
										}
									}
									modeltypetoken1[ival.(map[string]interface{})["precision"].(string)] = tokenlist1
									beammodel0[ival.(map[string]interface{})["case_name"].(string) + "-" + modelsize] = maplist
								}
								model1map[ival.(map[string]interface{})["case_name"].(string) + "-" + modelsize] = modeltypetoken1
							}




						}


					}else if sizenum > 13{
						if ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["beam_search"].(string) == "1" && ival.(map[string]interface{})["batch_size"].(string) == "1" {
							if checkDuplicateString(modellist2, ival.(map[string]interface{})["case_name"].(string) + "-" + modelsize){
								modellist2 = append(modellist2, ival.(map[string]interface{})["case_name"].(string) + "-" + modelsize)
								typelist2 := make([]string, 0)
								tokenlist2 := make([]string, 0)
								modeltypetoken2 := make(map[string][]string)
								if checkDuplicateString(typelist2, ival.(map[string]interface{})["precision"].(string)){
									typelist2 = append(typelist2, ival.(map[string]interface{})["precision"].(string))
									token := ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["input_tokens"].(string) + "/" + ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["max_new_tokens"].(string)
									// log.Println("token",token)
									maplist := make([]map[string]interface{},0)
									if checkDuplicateString(tokenlist2, token){
										tokenlist2 = append(tokenlist2, token)
										
										detailmap := make(map[string]interface{})
										detailmap["in_out_token"] = token
										detailmap["precision"] = ival.(map[string]interface{})["precision"].(string)
										detailmap["cards"] = ival.(map[string]interface{})["card"].(string)
										detailmap["next_avg"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["avg_latency"].(string) 
										detailmap["next_p90"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["p90_latency"].(string) 
										detailmap["total"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["latency"].(string)
										detailmap["first"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["first_latency"].(string) 								
										detailmap["target_p90_next"] = "/"
										detailmap["target_cards"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["target_card"].(string)
										detailmap["target_total"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["target_latency"].(string)
										detailmap["target_avg_next"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["target_avg_latency"].(string)
										detailmap["target_first"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["target_first_latency"].(string)

										maplist = append(maplist, detailmap)

									}
									modeltypetoken2[ival.(map[string]interface{})["precision"].(string)] = tokenlist2
									greedymodel1[ival.(map[string]interface{})["case_name"].(string) + "-" + modelsize] = maplist

								}else{
									tokenlist2 = modeltypetoken2[ival.(map[string]interface{})["precision"].(string)]
									maplist := greedymodel1[ival.(map[string]interface{})["case_name"].(string) + "-" + modelsize]
									token := ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["input_tokens"].(string) + "/" + ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["max_new_tokens"].(string)
									if checkDuplicateString(tokenlist2, token){
										tokenlist2 = append(tokenlist2, token)

										detailmap := make(map[string]interface{})
										detailmap["in_out_token"] = token
										detailmap["precision"] = ival.(map[string]interface{})["precision"].(string)
										detailmap["cards"] = ival.(map[string]interface{})["card"].(string)
										detailmap["next_avg"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["avg_latency"].(string) 
										detailmap["next_p90"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["p90_latency"].(string) 
										detailmap["total"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["latency"].(string)
										detailmap["first"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["first_latency"].(string) 								
										detailmap["target_p90_next"] = "/"
										detailmap["target_cards"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["target_card"].(string)
										detailmap["target_total"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["target_latency"].(string)
										detailmap["target_avg_next"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["target_avg_latency"].(string)
										detailmap["target_first"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["target_first_latency"].(string)

										maplist = append(maplist, detailmap)

									}

									modeltypetoken2[ival.(map[string]interface{})["precision"].(string)] = tokenlist2
									greedymodel1[ival.(map[string]interface{})["case_name"].(string) + "-" + modelsize] = maplist


								}
								model2map[ival.(map[string]interface{})["case_name"].(string) + "-" + modelsize] = modeltypetoken2


							}else{

								maplist := greedymodel1[ival.(map[string]interface{})["case_name"].(string) + "-" + modelsize]
								modeltypetoken2 := model2map[ival.(map[string]interface{})["case_name"].(string) + "-" + modelsize].(map[string][]string)

								if modeltypetoken2[ival.(map[string]interface{})["precision"].(string)] == nil{
									token := ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["input_tokens"].(string) + "/" + ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["max_new_tokens"].(string)
									tokenlist2 := modeltypetoken2[ival.(map[string]interface{})["precision"].(string)]
									if checkDuplicateString(tokenlist2, token){

										tokenlist2 = append(tokenlist2, token)
										detailmap := make(map[string]interface{})
										detailmap["in_out_token"] = token
										detailmap["precision"] = ival.(map[string]interface{})["precision"].(string)
										detailmap["cards"] = ival.(map[string]interface{})["card"].(string)
										detailmap["next_avg"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["avg_latency"].(string) 
										detailmap["next_p90"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["p90_latency"].(string) 
										detailmap["total"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["latency"].(string)
										detailmap["first"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["first_latency"].(string) 								
										detailmap["target_p90_next"] = "/"
										detailmap["target_cards"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["target_card"].(string)
										detailmap["target_total"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["target_latency"].(string)
										detailmap["target_avg_next"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["target_avg_latency"].(string)
										detailmap["target_first"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["target_first_latency"].(string)

										maplist = append(maplist, detailmap)

									}
									modeltypetoken2[ival.(map[string]interface{})["precision"].(string)] = tokenlist2
									greedymodel1[ival.(map[string]interface{})["case_name"].(string) + "-" + modelsize] = maplist

								}else{
									maplist := greedymodel1[ival.(map[string]interface{})["case_name"].(string) + "-" + modelsize]

									tokenlist2 := modeltypetoken2[ival.(map[string]interface{})["precision"].(string)]
									token := ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["input_tokens"].(string) + "/" + ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["max_new_tokens"].(string)

									if checkDuplicateString(tokenlist2, token){
										tokenlist2 = append(tokenlist2, token)
										detailmap := make(map[string]interface{})
										detailmap["in_out_token"] = token
										detailmap["precision"] = ival.(map[string]interface{})["precision"].(string)
										detailmap["cards"] = ival.(map[string]interface{})["card"].(string)
										detailmap["next_avg"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["avg_latency"].(string) 
										detailmap["next_p90"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["p90_latency"].(string) 
										detailmap["total"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["latency"].(string)
										detailmap["first"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["first_latency"].(string) 								
										detailmap["target_p90_next"] = "/"
										detailmap["target_cards"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["target_card"].(string)
										detailmap["target_total"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["target_latency"].(string)
										detailmap["target_avg_next"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["target_avg_latency"].(string)
										detailmap["target_first"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["target_first_latency"].(string)

										maplist = append(maplist, detailmap)
									}
									modeltypetoken2[ival.(map[string]interface{})["precision"].(string)] = tokenlist2
									greedymodel1[ival.(map[string]interface{})["case_name"].(string) + "-" + modelsize] = maplist

								}
								model2map[ival.(map[string]interface{})["case_name"].(string) + "-" + modelsize] = modeltypetoken2

							}

						}else if ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["beam_search"].(string) == "4" && ival.(map[string]interface{})["batch_size"].(string) == "1" {
							if checkDuplicateString(modellist1, ival.(map[string]interface{})["case_name"].(string) + "-" + modelsize){
								modellist3 = append(modellist3, ival.(map[string]interface{})["case_name"].(string) + "-" + modelsize)
								typelist3 := make([]string, 0)
								tokenlist3 := make([]string, 0)
								modeltypetoken3 := make(map[string][]string)
								if checkDuplicateString(typelist3, ival.(map[string]interface{})["precision"].(string)){
									typelist3 = append(typelist3, ival.(map[string]interface{})["precision"].(string))
									token := ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["input_tokens"].(string) + "/" + ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["max_new_tokens"].(string)
									maplist := make([]map[string]interface{},0)
									if checkDuplicateString(tokenlist3, token){
										tokenlist3 = append(tokenlist3, token)
										
										detailmap := make(map[string]interface{})
										detailmap["in_out_token"] = token
										detailmap["precision"] = ival.(map[string]interface{})["precision"].(string)
										detailmap["cards"] = ival.(map[string]interface{})["card"].(string)
										detailmap["next_avg"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["avg_latency"].(string) 
										detailmap["next_p90"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["p90_latency"].(string) 
										detailmap["total"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["latency"].(string)
										detailmap["first"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["first_latency"].(string) 								
										detailmap["target_p90_next"] = "/"
										detailmap["target_cards"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["target_card"].(string)
										detailmap["target_total"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["target_latency"].(string)
										detailmap["target_avg_next"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["target_avg_latency"].(string)
										detailmap["target_first"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["target_first_latency"].(string)

										maplist = append(maplist, detailmap)

									}
									modeltypetoken3[ival.(map[string]interface{})["precision"].(string)] = tokenlist3
									beammodel1[ival.(map[string]interface{})["case_name"].(string) + "-" + modelsize] = maplist

								}else{

									tokenlist3 = modeltypetoken3[ival.(map[string]interface{})["precision"].(string)]
									maplist := beammodel1[ival.(map[string]interface{})["case_name"].(string) + "-" + modelsize]
									token := ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["input_tokens"].(string) + "/" + ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["max_new_tokens"].(string)
									if checkDuplicateString(tokenlist3, token){

										tokenlist3 = append(tokenlist3, token)
										
										detailmap := make(map[string]interface{})
										detailmap["in_out_token"] = token
										detailmap["precision"] = ival.(map[string]interface{})["precision"].(string)
										detailmap["cards"] = ival.(map[string]interface{})["card"].(string)
										detailmap["next_avg"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["avg_latency"].(string) 
										detailmap["next_p90"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["p90_latency"].(string) 
										detailmap["total"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["latency"].(string)
										detailmap["first"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["first_latency"].(string) 								
										detailmap["target_p90_next"] = "/"
										detailmap["target_cards"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["target_card"].(string)
										detailmap["target_total"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["target_latency"].(string)
										detailmap["target_avg_next"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["target_avg_latency"].(string)
										detailmap["target_first"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["target_first_latency"].(string)

										maplist = append(maplist, detailmap)

									}
									modeltypetoken3[ival.(map[string]interface{})["precision"].(string)] = tokenlist3
									beammodel1[ival.(map[string]interface{})["case_name"].(string) + "-" + modelsize] = maplist

								}

								model3map[ival.(map[string]interface{})["case_name"].(string) + "-" + modelsize] = modeltypetoken3


							}else{

								maplist := beammodel1[ival.(map[string]interface{})["case_name"].(string) + "-" + modelsize]
								modeltypetoken3 := model3map[ival.(map[string]interface{})["case_name"].(string) + "-" + modelsize].(map[string][]string)

								if modeltypetoken3[ival.(map[string]interface{})["precision"].(string)] == nil{
									token := ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["input_tokens"].(string) + "/" + ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["max_new_tokens"].(string)
									tokenlist3 := modeltypetoken3[ival.(map[string]interface{})["precision"].(string)]
									if checkDuplicateString(tokenlist3, token){
										tokenlist3 = append(tokenlist3, token)
											
										detailmap := make(map[string]interface{})
										detailmap["in_out_token"] = token
										detailmap["precision"] = ival.(map[string]interface{})["precision"].(string)
										detailmap["cards"] = ival.(map[string]interface{})["card"].(string)
										detailmap["next_avg"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["avg_latency"].(string) 
										detailmap["next_p90"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["p90_latency"].(string) 
										detailmap["total"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["latency"].(string)
										detailmap["first"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["first_latency"].(string) 								
										detailmap["target_p90_next"] = "/"
										detailmap["target_cards"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["target_card"].(string)
										detailmap["target_total"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["target_latency"].(string)
										detailmap["target_avg_next"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["target_avg_latency"].(string)
										detailmap["target_first"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["target_first_latency"].(string)

										maplist = append(maplist, detailmap)


									}
									modeltypetoken3[ival.(map[string]interface{})["precision"].(string)] = tokenlist3
									beammodel1[ival.(map[string]interface{})["case_name"].(string) + "-" + modelsize] = maplist
								}else{

									maplist := beammodel1[ival.(map[string]interface{})["case_name"].(string) + "-" + modelsize]
									tokenlist3 := modeltypetoken3[ival.(map[string]interface{})["precision"].(string)]
									token := ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["input_tokens"].(string) + "/" + ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["max_new_tokens"].(string)
									if checkDuplicateString(tokenlist3, token){

										tokenlist3 = append(tokenlist3, token)
											
										detailmap := make(map[string]interface{})
										detailmap["in_out_token"] = token
										detailmap["precision"] = ival.(map[string]interface{})["precision"].(string)
										detailmap["cards"] = ival.(map[string]interface{})["card"].(string)
										detailmap["next_avg"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["avg_latency"].(string) 
										detailmap["next_p90"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["p90_latency"].(string) 
										detailmap["total"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["latency"].(string)
										detailmap["first"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["first_latency"].(string) 								
										detailmap["target_p90_next"] = "/"
										detailmap["target_cards"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["target_card"].(string)
										detailmap["target_total"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["target_latency"].(string)
										detailmap["target_avg_next"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["target_avg_latency"].(string)
										detailmap["target_first"] = ival.(map[string]interface{})["anyinterface"].(map[string]interface{})["target_first_latency"].(string)

										maplist = append(maplist, detailmap)



									}
									modeltypetoken3[ival.(map[string]interface{})["precision"].(string)] = tokenlist3
									beammodel1[ival.(map[string]interface{})["case_name"].(string) + "-" + modelsize] = maplist
								}
								model3map[ival.(map[string]interface{})["case_name"].(string) + "-" + modelsize] = modeltypetoken3
							}

						}
					}

					
				}

				// for training part 
				// 	for _, ival := range trainset["data"].([]interface{}){

				// 		if checkDuplicateString(modellist4, ival.(map[string]interface{})["case_name"].(string)){
				// 			modellist4 = append(modellist4, ival.(map[string]interface{})["case_name"].(string))
				// 			typelist4 := make([]string, 0)
				// 			configlist4 := make([]string, 0)
				// 			modeltypeconfig4 := make(map[string][]string)
				// 			if checkDuplicateString(typelist4, ival.(map[string]interface{})["precision"].(string)){
				// 				typelist4 = append(typelist4, ival.(map[string]interface{})["precision"].(string))
				// 				// "BS=1 ZeRo=1 PP=5 MP=1 DP=12 GAS=10 GLOBAL_BS=120"
				// 				config := "BS=" + ival.(map[string]interface{})["case_name"].(string)



				// 			}
				// 		}


				// 	}

				// log.Println("modellist0", modellist0)
				// log.Println("model0map", model0map)
				// model0map = model0json

				//for config
				for onetype, oneval := range sureSet["configurediff"].(map[string]interface{}){
					if strings.ToLower(strings.Split(onetype, "_")[2]) == "realtime"{
						infercon = oneval.(map[string]interface{})
					}else if strings.ToLower(strings.Split(onetype, "_")[2]) == "training"{
						traincon = oneval.(map[string]interface{})
					}
				}



				if len(infercon)>0{
					inferconsw := make(map[string]interface{})
					inferconhw := make(map[string]interface{})
					inferconmap := make(map[string]interface{})
					for onekey, oneval := range infercon{
						switch onekey{
							case "pytorch":
								keyconfig := "pytorch "
								for ikey, ival := range oneval.(map[string]interface{}){
									if ikey == "branch"{
										keyconfig += ival.(string)
										keyconfig += " @ "
									}
								}
								for ikey, ival := range oneval.(map[string]interface{}){
									if ikey == "commit_id"{
										keyconfig += ival.(string)
									}
								}
								inferconsw["PyTorch"] = keyconfig

							case "ipex":
								keyconfig := "IPEX "
								for ikey, ival := range oneval.(map[string]interface{}){
									if ikey == "branch"{
										keyconfig += ival.(string) 
										keyconfig += " @ "
									}
									if ikey == "commit_id" {
										keyconfig += ival.(string)
									}
								}
								inferconsw["IPEX"] = keyconfig
							
							case "compiler":
								inferconsw["Compiler"] = oneval.(string)
							case "driver":
								inferconsw["Driver"] = oneval.(string)
							
							
						}
					}
					inferconsw["Transformer"] = "4.31.0"

					inferconhw["Host"] = "-"
					inferconhw["PVC"] = "600W OAM card, 512EU/tile"

					inferconmap["swconfig"] = inferconsw
					inferconmap["hwconfig"] = inferconhw

					configmap["Inference"] = inferconmap
				}

				if len(traincon)>0{
					trainconsw := make(map[string]interface{})
					trainconhw := make(map[string]interface{})
					trainconmap := make(map[string]interface{})
					for onekey, oneval := range traincon{
						switch onekey{
							case "pytorch":
								keyconfig := "pytorch "
								for ikey, ival := range oneval.(map[string]interface{}){
									if ikey == "branch"{
										keyconfig += ival.(string)
										keyconfig += " @ "
									}
								}
								for ikey, ival := range oneval.(map[string]interface{}){
									if ikey == "commit_id"{
										keyconfig += ival.(string)
									}
								}
								trainconsw["PyTorch"] = keyconfig

							case "ipex":
								keyconfig := "IPEX "
								for ikey, ival := range oneval.(map[string]interface{}){
									if ikey == "branch"{
										keyconfig += ival.(string) 
										keyconfig += " @ "
									}
									if ikey == "commit_id" {
										keyconfig += ival.(string)
									}
								}
								trainconsw["IPEX"] = keyconfig
							
							case "compiler":
								trainconsw["Compiler"] = oneval
							case "driver":
								trainconsw["Driver"] = oneval
							
							
						}
					}
					trainconsw["Transformer"] = "4.31.0"

					trainconhw["Cluster"] = "-"
					trainconhw["PVC"] = "450W OAM, 448EU/tile"

					trainconmap["swconfig"] = trainconsw
					trainconmap["hwconfig"] = trainconhw

					configmap["Training"] = trainconmap
				}


				littlemodelmap["Greedy"] = model0map
				littlemodelmap["BeamSearch4"] = model1map
				
				littlemodeljson := make(map[string]interface{})
				bigmodeljson := make(map[string]interface{})
				littlemodeljson["Greedy"] = Grafanalayer(greedymodel0)
				littlemodeljson["BeamSearch4"] = Grafanalayer(beammodel0)
				bigmodeljson["Greedy"] = Grafanalayer(greedymodel1)
				bigmodeljson["BeamSearch4"] = Grafanalayer(beammodel1)
				model0json["1B-13B"] = littlemodeljson
				model0json["30B-176B"] = bigmodeljson




			
				jsonmap[hardware + "LLM"] = model0json
				jsonmap[hardware + "LLMConfig"] = configmap
			}else{
				jsonmap["log"] = "Sorry, there is no data for this date."
			}


		case "kpi":
			if processor == "gpu"{
				databaseipex := "ipex_gpu_v3_verified_transfer"
				databaseitex := "itex_gpu_v3_verified_transfer"

				vtipex := session.DB(useBase).C(databaseipex)
				vtitex := session.DB(useBase).C(databaseitex)

				dataSet1 := make([]bson.M, 0)
				dataSet2 := make([]bson.M, 0)
				dataSet3 := make([]bson.M, 0)
				dataSet4 := make([]bson.M, 0)

				var periodlist1, periodlist2, periodlist3, periodlist4 []string

				sureSet1 := make(map[string]interface{})
				sureSet2 := make(map[string]interface{})
				sureSet3 := make(map[string]interface{})
				sureSet4 := make(map[string]interface{})

				kpiPVCquery := bson.M{"hardware":"PVC","period":bson.M{"$regex": workweek}}
				kpiATSMquery := bson.M{"hardware":"ATSM","period":bson.M{"$regex": workweek}}

				kpiIPEXerr1 := vtipex.Find(kpiPVCquery).All(&dataSet1)
				kpiIPEXerr2 := vtipex.Find(kpiATSMquery).All(&dataSet2)
				kpiIPEXerr3 := vtitex.Find(kpiPVCquery).All(&dataSet3)
				kpiIPEXerr4 := vtitex.Find(kpiATSMquery).All(&dataSet4)
				
				jsonmap["week"] = workweek
				hwmap := make(map[string]interface{})
				//ITEX PVC
				if kpiIPEXerr3 == nil && len(dataSet3) > 0{
					for _, oneJson := range dataSet3{
						// log.Println("period",oneJson["period"].(string))
						if strings.Split(oneJson["period"].(string), ".")[1] != "9"{
							periodlist3 = append(periodlist3, oneJson["period"].(string))
						}
	
					}
					var selectedperiod string
					if len(periodlist3) > 0{
						sort.Strings(periodlist3)
						selectedperiod = periodlist3[len(periodlist3)-1]
						sureQuery := bson.M{"hardware":"PVC","period":selectedperiod}
						surerr := vtitex.Find(sureQuery).One(&sureSet3)
						log.Println("surerr3",surerr, selectedperiod)

						// itexpvcpart := KpiITEX("PVC","itex","gpu" ,sureSet3)
						hwmap["PVC"] = KpiITEX("PVC", "itex", "gpu", sureSet3)
						jsonmap["GPUDetails"] = hwmap


					}
				}else{
					//jsonmap["ITEX PVC"] = "Sorry, there is no data for this date."
					log.Println("ITEX PVC: Sorry, there is no data for this date.")
				}

				//ITEX ATSM
				if kpiIPEXerr4 == nil && len(dataSet4) > 0{
					for _, oneJson := range dataSet4{
						// log.Println("period",oneJson["period"].(string))
						if strings.Split(oneJson["period"].(string), ".")[1] != "9"{
							periodlist4 = append(periodlist4, oneJson["period"].(string))
						}
	
					}
					var selectedperiod string
					if len(periodlist4) > 0{
						sort.Strings(periodlist4)
						selectedperiod = periodlist4[len(periodlist4)-1]
						sureQuery := bson.M{"hardware":"ATSM","period":selectedperiod}
						surerr := vtitex.Find(sureQuery).One(&sureSet4)
						log.Println("surerr4",surerr, selectedperiod)

						hwmap["ATSM"] = KpiITEX("ATSM", "itex", "gpu", sureSet4)
						jsonmap["GPUDetails"] = hwmap


					}
				}else{
					// jsonmap["ITEX ATSM"] = "Sorry, there is no data for this date."
					log.Println("ITEX ATSM: Sorry, there is no data for this date.")
				}


				//IPEX PVC
				if kpiIPEXerr1 == nil && len(dataSet1) > 0{  
					for _, oneJson := range dataSet1{
						// log.Println("period",oneJson["period"].(string))
						if strings.Split(oneJson["period"].(string), ".")[1] != "9"{
							periodlist1 = append(periodlist1, oneJson["period"].(string))
						}
	
					}

					var selectedperiod string
					if len(periodlist1) > 0{
						sort.Strings(periodlist1)
						selectedperiod = periodlist1[len(periodlist1)-1]
						sureQuery := bson.M{"hardware":"PVC","period":selectedperiod}
						surerr := vtipex.Find(sureQuery).One(&sureSet1)
						log.Println("surerr1",surerr, selectedperiod)

						

					}

				}else{
					// jsonmap["IPEX PVC"] = "Sorry, there is no data for this date."
					log.Println("IPEX PVC: Sorry, there is no data for this date.")
				}

				//IPEX ATSM
				if kpiIPEXerr2 == nil && len(dataSet2) > 0{
					for _, oneJson := range dataSet2{
						// log.Println("period",oneJson["period"].(string))
						if strings.Split(oneJson["period"].(string), ".")[1] != "9"{
							periodlist2 = append(periodlist2, oneJson["period"].(string))
						}
	
					}
	
					var selectedperiod string
					if len(periodlist2) > 0{
						sort.Strings(periodlist2)
						selectedperiod = periodlist2[len(periodlist2)-1]
						sureQuery := bson.M{"hardware":"ATSM","period":selectedperiod}
						surerr := vtipex.Find(sureQuery).One(&sureSet2)
						log.Println("surerr2",surerr, selectedperiod)
					}
				}else{
					jsonmap["IPEX ATSM"] = "Sorry, there is no data for this date."
				}





			}

	}


	return jsonmap
}

func Grafanalayer(layermap map[string][]map[string]interface {})map[string]interface{}{

	// log.Println("layermap",layermap)
	modellayer := make(map[string]interface{})
	for modelid, ival := range layermap{
		tokenlist := make([]string, 0)
		
		modeltoken := make(map[string]interface{})
		for _, oneiter := range ival{
			
			if checkDuplicateString(tokenlist, oneiter["in_out_token"].(string)){
				typelist := make([]string, 0)
				onetoken := oneiter["in_out_token"].(string)
				tokenlist = append(tokenlist, oneiter["in_out_token"].(string))
				modeltype := make(map[string]interface{})
				
				if checkDuplicateString(typelist, oneiter["precision"].(string)){
					onetype := oneiter["precision"].(string)
					typelist = append(typelist, oneiter["precision"].(string))
					delete(oneiter, "precision")
					delete(oneiter, "in_out_token")
					modeltype[onetype] = oneiter
				}
				
				modeltoken[onetoken] = modeltype
			}else{

				usedmap := modeltoken[oneiter["in_out_token"].(string)].(map[string]interface{})
				onetoken := oneiter["in_out_token"].(string)
				if usedmap[oneiter["precision"].(string)] == nil{
					onetype := oneiter["precision"].(string)
					delete(oneiter, "precision")
					delete(oneiter, "in_out_token")
					usedmap[onetype] = oneiter

				}
				modeltoken[onetoken] = usedmap
			}
			
		}
		modellayer[modelid] = modeltoken
		// ival = modellayer
	}
	
	return modellayer
}

func KpiITEX(hardware, framework, processor string, dataSet map[string]interface{})map[string]interface{}{
	session, useBase := judgeDatabase()
	defer session.Close()
	database := framework + "_" + processor + "_competitor"

	resultmap := make(map[string]interface{})

	c := session.DB(useBase).C(database)
	var dateinterface []interface{}
	var newtime int64
	pipeline := []bson.M{
		{"$match":bson.M{"category":"competitor", "hardware":hardware}},
		{"$project":bson.M{"time":1, "_id":0}},
	}
	pipe := c.Pipe(pipeline)
	errpipr := pipe.AllowDiskUse().All(&dateinterface)

	switch hardware{
		case "PVC":

			if errpipr == nil && len(dateinterface) > 0{
				por_1t_infer := make([]interface{},0)
				por_1t_train := make([]interface{},0)
				por_nt_infer := make([]interface{},0)
				por_nt_train := make([]interface{},0)
				por_hvd := make([]interface{},0)

				por_1t_infer_used := make([]interface{},0)
				por_1t_train_used := make([]interface{},0)
				por_nt_infer_used := make([]interface{},0)
				por_nt_train_used := make([]interface{},0)
				por_hvd_used := make([]interface{},0)

				for _, oneGe := range dataSet["general"].([]interface{}){
					switch oneGe.(map[string]interface{})["case_suite"].(string){
						case "HVD-POR-NT":
							por_hvd = oneGe.(map[string]interface{})["data"].([]interface{})
						case "POR-NT":
							switch strings.ToLower(strings.Split(oneGe.(map[string]interface{})["r2_type"].(string), "@")[0]){
								case "training":
									por_nt_train = oneGe.(map[string]interface{})["data"].([]interface{})
								case "inference":
									por_nt_infer = oneGe.(map[string]interface{})["data"].([]interface{})
							}
						case "POR-1T":
							switch strings.ToLower(strings.Split(oneGe.(map[string]interface{})["r2_type"].(string), "@")[0]){
							case "training":
								por_1t_train = oneGe.(map[string]interface{})["data"].([]interface{})
							case "inference":
								por_1t_infer = oneGe.(map[string]interface{})["data"].([]interface{})
						}

					}
				}


				for _, onetime := range dateinterface{

					if onetime.(bson.M)["time"].(int64) > newtime {
						newtime = onetime.(bson.M)["time"].(int64)  //latest
					}
				}

				competitorbson := make(map[string]interface{})
				queryBson := bson.M{"time":newtime, "hardware":hardware}
				err := c.Find(queryBson).One(&competitorbson)

				if err == nil{
					plist := make([]string, 0)
					typelist := make([]string, 0)
					modellist := make([]string, 0)


					priomap := make(map[string]interface{})
					framemap := make(map[string]interface{})
					runmap := make(map[string]interface{})

					// framemap["Tensorflow"] = priomap
					// runmap["inference"] = framemap

					if len(por_1t_infer) > 0{
						for _, oneData := range por_1t_infer{
							keyname := oneData.(map[string]interface{})["test_mode"].(string) + " " + oneData.(map[string]interface{})["case_name"].(string) + " " + 
								oneData.(map[string]interface{})["layout"].(string) + " " + oneData.(map[string]interface{})["precision"].(string) + " " + 
								oneData.(map[string]interface{})["run_type"].(string) + " " + oneData.(map[string]interface{})["batch_size"].(string) + " " + 
								oneData.(map[string]interface{})["scenario"].(string)

							for _, onecom := range competitorbson["competitor"].([]interface{}){
								if keyname == onecom.(map[string]interface{})["key_name"]{
									oneData.(map[string]interface{})["priority"] = strings.ToUpper(onecom.(map[string]interface{})["priority"].(string))
									oneData.(map[string]interface{})["competitor"] = onecom.(map[string]interface{})["competitor_number"]
									oneData.(map[string]interface{})["ratio"] = onecom.(map[string]interface{})["pri-multiplier"]

									por_1t_infer_used = append(por_1t_infer_used, oneData)
				
								}
							}
						}
						
					}

					if len(por_1t_train) > 0{
						for _, oneData := range por_1t_train{
							keyname := oneData.(map[string]interface{})["test_mode"].(string) + " " + oneData.(map[string]interface{})["case_name"].(string) + " " + 
								oneData.(map[string]interface{})["layout"].(string) + " " + oneData.(map[string]interface{})["precision"].(string) + " " + 
								oneData.(map[string]interface{})["run_type"].(string) + " " + oneData.(map[string]interface{})["batch_size"].(string) + " " + 
								oneData.(map[string]interface{})["scenario"].(string)

							for _, onecom := range competitorbson["competitor"].([]interface{}){
								if keyname == onecom.(map[string]interface{})["key_name"]{
									oneData.(map[string]interface{})["priority"] = strings.ToUpper(onecom.(map[string]interface{})["priority"].(string))
									oneData.(map[string]interface{})["competitor"] = onecom.(map[string]interface{})["competitor_number"]
									oneData.(map[string]interface{})["ratio"] = onecom.(map[string]interface{})["pri-multiplier"]

									por_1t_train_used = append(por_1t_train_used, oneData)
				
								}
							}
						}
						
					}

					if len(por_nt_infer) > 0{
						for _, oneData := range por_nt_infer{
							keyname := oneData.(map[string]interface{})["test_mode"].(string) + " " + oneData.(map[string]interface{})["case_name"].(string) + " " + 
								oneData.(map[string]interface{})["layout"].(string) + " " + oneData.(map[string]interface{})["precision"].(string) + " " + 
								oneData.(map[string]interface{})["run_type"].(string) + " " + oneData.(map[string]interface{})["batch_size"].(string) + " " + 
								oneData.(map[string]interface{})["scenario"].(string)

							for _, onecom := range competitorbson["competitor"].([]interface{}){
								if keyname == onecom.(map[string]interface{})["key_name"]{
									oneData.(map[string]interface{})["priority"] = strings.ToUpper(onecom.(map[string]interface{})["priority"].(string))
									oneData.(map[string]interface{})["competitor"] = onecom.(map[string]interface{})["competitor_number"]
									oneData.(map[string]interface{})["ratio"] = onecom.(map[string]interface{})["pri-multiplier"]

									por_nt_infer_used = append(por_nt_infer_used, oneData)
				
								}
							}
						}
						
					}

					if len(por_nt_train) > 0{
						for _, oneData := range por_nt_train{
							keyname := oneData.(map[string]interface{})["test_mode"].(string) + " " + oneData.(map[string]interface{})["case_name"].(string) + " " + 
								oneData.(map[string]interface{})["layout"].(string) + " " + oneData.(map[string]interface{})["precision"].(string) + " " + 
								oneData.(map[string]interface{})["run_type"].(string) + " " + oneData.(map[string]interface{})["batch_size"].(string) + " " + 
								oneData.(map[string]interface{})["scenario"].(string)

							for _, onecom := range competitorbson["competitor"].([]interface{}){
								if keyname == onecom.(map[string]interface{})["key_name"]{
									oneData.(map[string]interface{})["priority"] = strings.ToUpper(onecom.(map[string]interface{})["priority"].(string))
									oneData.(map[string]interface{})["competitor"] = onecom.(map[string]interface{})["competitor_number"]
									oneData.(map[string]interface{})["ratio"] = onecom.(map[string]interface{})["pri-multiplier"]

									por_nt_train_used = append(por_nt_train_used, oneData)
				
								}
							}
						}
						
					}

					if len(por_hvd) > 0{
						for _, oneData := range por_hvd{
							keyname := oneData.(map[string]interface{})["test_mode"].(string) + " " + oneData.(map[string]interface{})["case_name"].(string) + " " + 
								oneData.(map[string]interface{})["layout"].(string) + " " + oneData.(map[string]interface{})["precision"].(string) + " " + 
								oneData.(map[string]interface{})["run_type"].(string) + " " + oneData.(map[string]interface{})["batch_size"].(string) + " " + 
								oneData.(map[string]interface{})["scenario"].(string)

							for _, onecom := range competitorbson["competitor"].([]interface{}){
								if keyname == onecom.(map[string]interface{})["key_name"]{
									oneData.(map[string]interface{})["priority"] = strings.ToUpper(onecom.(map[string]interface{})["priority"].(string))
									oneData.(map[string]interface{})["competitor"] = onecom.(map[string]interface{})["competitor_number"]
									oneData.(map[string]interface{})["ratio"] = onecom.(map[string]interface{})["pri-multiplier"]

									por_hvd_used = append(por_hvd_used, oneData)
				
								}
							}
						}
						
					}


					for _, oneiter := range por_1t_infer_used{
						
						if checkDuplicateString(plist, oneiter.(map[string]interface{})["priority"].(string)){
							// log.Println("PPPP",oneiter.(map[string]interface{})["priority"].(string))
							plist = append(plist, oneiter.(map[string]interface{})["priority"].(string))
							if checkDuplicateString(typelist, oneiter.(map[string]interface{})["precision"].(string)){
								typelist = append(typelist, oneiter.(map[string]interface{})["precision"].(string))
								
								if checkDuplicateString(modellist, oneiter.(map[string]interface{})["case_name"].(string)){

									modellist = append(modellist, oneiter.(map[string]interface{})["case_name"].(string))

									modelmap := make(map[string]interface{})
									kindmap := make(map[string]interface{})
									typemap := make(map[string]interface{})
									onemap := make(map[string]interface{})
									
									onemap["perf"] = oneiter.(map[string]interface{})["value"].(string)
									onemap["competitor"] = oneiter.(map[string]interface{})["competitor"].(string)

									if StringTransferToFloat64(oneiter.(map[string]interface{})["value"].(string)) > 0 && StringTransferToFloat64(oneiter.(map[string]interface{})["competitor"].(string)) > 0{
										onemap["sore"] = fmt.Sprintf("%.4f", (StringTransferToFloat64(oneiter.(map[string]interface{})["value"].(string))/StringTransferToFloat64(oneiter.(map[string]interface{})["competitor"].(string))))
									}else{
										onemap["sore"] = "/"
									}
			
									if StringTransferToFloat64(onemap["sore"].(string)) > 0 && StringTransferToFloat64(oneiter.(map[string]interface{})["ratio"].(string)) > 0{
										onemap["goalscore"] = fmt.Sprintf("%.4f", (StringTransferToFloat64(onemap["sore"].(string))/StringTransferToFloat64(oneiter.(map[string]interface{})["ratio"].(string))))
									}else{
										onemap["goalscore"] = "/"
									}
			
									if StringTransferToFloat64(oneiter.(map[string]interface{})["competitor"].(string)) <= 0 && StringTransferToFloat64(onemap["perf"].(string)) > 0{
										onemap["valuetag"] = "noproject"
									}else if StringTransferToFloat64(oneiter.(map[string]interface{})["competitor"].(string)) > 0 && StringTransferToFloat64(onemap["perf"].(string)) <= 0{
										onemap["valuetag"] = "novalue"
									}else if StringTransferToFloat64(oneiter.(map[string]interface{})["competitor"].(string)) > 0 && StringTransferToFloat64(onemap["perf"].(string)) > 0{
										onemap["valuetag"] = "normal"
									}else{
										onemap["value"] = "/"
									}
									
									
									modelmap[oneiter.(map[string]interface{})["case_name"].(string)] = onemap
									kindmap["Batched inference"] = modelmap
									typemap[oneiter.(map[string]interface{})["precision"].(string)] = kindmap
									priomap[oneiter.(map[string]interface{})["priority"].(string)] = typemap

								}

								
								
								
								
							}else{
								// log.Println("来过这吗？")
								onetypemap := priomap[oneiter.(map[string]interface{})["priority"].(string)]
								onekindmap := onetypemap.(map[string]interface{})[oneiter.(map[string]interface{})["precision"].(string)]
								onemodelmap := onekindmap.(map[string]interface{})["Batched inference"]

								if onemodelmap.(map[string]interface{})[oneiter.(map[string]interface{})["case_name"].(string)] == nil{
									onemap := make(map[string]interface{})
									onemap["perf"] = oneiter.(map[string]interface{})["value"].(string)
									onemap["competitor"] = oneiter.(map[string]interface{})["competitor"].(string)

									if StringTransferToFloat64(oneiter.(map[string]interface{})["value"].(string)) > 0 && StringTransferToFloat64(oneiter.(map[string]interface{})["competitor"].(string)) > 0{
										onemap["sore"] = fmt.Sprintf("%.4f", (StringTransferToFloat64(oneiter.(map[string]interface{})["value"].(string))/StringTransferToFloat64(oneiter.(map[string]interface{})["competitor"].(string))))
									}else{
										onemap["sore"] = "/"
									}
			
									if StringTransferToFloat64(onemap["sore"].(string)) > 0 && StringTransferToFloat64(oneiter.(map[string]interface{})["ratio"].(string)) > 0{
										onemap["goalscore"] = fmt.Sprintf("%.4f", (StringTransferToFloat64(onemap["sore"].(string))/StringTransferToFloat64(oneiter.(map[string]interface{})["ratio"].(string))))
									}else{
										onemap["goalscore"] = "/"
									}
			
									if StringTransferToFloat64(oneiter.(map[string]interface{})["competitor"].(string)) <= 0 && StringTransferToFloat64(onemap["perf"].(string)) > 0{
										onemap["valuetag"] = "noproject"
									}else if StringTransferToFloat64(oneiter.(map[string]interface{})["competitor"].(string)) > 0 && StringTransferToFloat64(onemap["perf"].(string)) <= 0{
										onemap["valuetag"] = "novalue"
									}else if StringTransferToFloat64(oneiter.(map[string]interface{})["competitor"].(string)) > 0 && StringTransferToFloat64(onemap["perf"].(string)) > 0{
										onemap["valuetag"] = "normal"
									}else{
										onemap["value"] = "/"
									}

									onemodelmap.(map[string]interface{})[oneiter.(map[string]interface{})["case_name"].(string)] = onemap
								}
								onekindmap.(map[string]interface{})["Batched inference"] = onemodelmap
								onetypemap.(map[string]interface{})[oneiter.(map[string]interface{})["precision"].(string)] = onekindmap
								priomap[oneiter.(map[string]interface{})["priority"].(string)] = onetypemap

							}


						}else{

							// log.Println("111 来过这吗？")
							onetypemap := priomap[oneiter.(map[string]interface{})["priority"].(string)]
							if onetypemap.(map[string]interface{})[oneiter.(map[string]interface{})["precision"].(string)] == nil {
								// log.Println("222 来过这吗？")
								kindmap2 := make(map[string]interface{})
								modelmap2 := make(map[string]interface{})
								onemap := make(map[string]interface{})
								onemap["perf"] = oneiter.(map[string]interface{})["value"].(string)
								onemap["competitor"] = oneiter.(map[string]interface{})["competitor"].(string)

								if StringTransferToFloat64(oneiter.(map[string]interface{})["value"].(string)) > 0 && StringTransferToFloat64(oneiter.(map[string]interface{})["competitor"].(string)) > 0{
									onemap["sore"] = fmt.Sprintf("%.4f", (StringTransferToFloat64(oneiter.(map[string]interface{})["value"].(string))/StringTransferToFloat64(oneiter.(map[string]interface{})["competitor"].(string))))
								}else{
									onemap["sore"] = "/"
								}

								if StringTransferToFloat64(onemap["sore"].(string)) > 0 && StringTransferToFloat64(oneiter.(map[string]interface{})["ratio"].(string)) > 0{
									onemap["goalscore"] = fmt.Sprintf("%.4f", (StringTransferToFloat64(onemap["sore"].(string))/StringTransferToFloat64(oneiter.(map[string]interface{})["ratio"].(string))))
								}else{
									onemap["goalscore"] = "/"
								}

								if StringTransferToFloat64(oneiter.(map[string]interface{})["competitor"].(string)) <= 0 && StringTransferToFloat64(onemap["perf"].(string)) > 0{
									onemap["valuetag"] = "noproject"
								}else if StringTransferToFloat64(oneiter.(map[string]interface{})["competitor"].(string)) > 0 && StringTransferToFloat64(onemap["perf"].(string)) <= 0{
									onemap["valuetag"] = "novalue"
								}else if StringTransferToFloat64(oneiter.(map[string]interface{})["competitor"].(string)) > 0 && StringTransferToFloat64(onemap["perf"].(string)) > 0{
									onemap["valuetag"] = "normal"
								}else{
									onemap["value"] = "/"
								}

								
								modelmap2[oneiter.(map[string]interface{})["case_name"].(string)] = onemap
								kindmap2["Batched inference"] = modelmap2
								onetypemap.(map[string]interface{})[oneiter.(map[string]interface{})["precision"].(string)] = kindmap2
								
							}else{
								// log.Println("333 来过这吗？")

								onekindmap := onetypemap.(map[string]interface{})[oneiter.(map[string]interface{})["precision"].(string)]
								onemodelmap := onekindmap.(map[string]interface{})["Batched inference"]

								if onemodelmap.(map[string]interface{})[oneiter.(map[string]interface{})["case_name"].(string)] == nil{
									// log.Println("444 来过这吗？")
									onemap := make(map[string]interface{})
									onemap["perf"] = oneiter.(map[string]interface{})["value"].(string)
									onemap["competitor"] = oneiter.(map[string]interface{})["competitor"].(string)

									if StringTransferToFloat64(oneiter.(map[string]interface{})["value"].(string)) > 0 && StringTransferToFloat64(oneiter.(map[string]interface{})["competitor"].(string)) > 0{
										onemap["sore"] = fmt.Sprintf("%.4f", (StringTransferToFloat64(oneiter.(map[string]interface{})["value"].(string))/StringTransferToFloat64(oneiter.(map[string]interface{})["competitor"].(string))))
									}else{
										onemap["sore"] = "/"
									}
			
									if StringTransferToFloat64(onemap["sore"].(string)) > 0 && StringTransferToFloat64(oneiter.(map[string]interface{})["ratio"].(string)) > 0{
										onemap["goalscore"] = fmt.Sprintf("%.4f", (StringTransferToFloat64(onemap["sore"].(string))/StringTransferToFloat64(oneiter.(map[string]interface{})["ratio"].(string))))
									}else{
										onemap["goalscore"] = "/"
									}
			
									if StringTransferToFloat64(oneiter.(map[string]interface{})["competitor"].(string)) <= 0 && StringTransferToFloat64(onemap["perf"].(string)) > 0{
										onemap["valuetag"] = "noproject"
									}else if StringTransferToFloat64(oneiter.(map[string]interface{})["competitor"].(string)) > 0 && StringTransferToFloat64(onemap["perf"].(string)) <= 0{
										onemap["valuetag"] = "novalue"
									}else if StringTransferToFloat64(oneiter.(map[string]interface{})["competitor"].(string)) > 0 && StringTransferToFloat64(onemap["perf"].(string)) > 0{
										onemap["valuetag"] = "normal"
									}else{
										onemap["value"] = "/"
									}

									onemodelmap.(map[string]interface{})[oneiter.(map[string]interface{})["case_name"].(string)] = onemap
								}
								onekindmap.(map[string]interface{})["Batched inference"] = onemodelmap
								onetypemap.(map[string]interface{})[oneiter.(map[string]interface{})["precision"].(string)] = onekindmap
								
							}


							priomap[oneiter.(map[string]interface{})["priority"].(string)] = onetypemap


						}
					}

					for _, oneiter := range por_nt_infer_used{
						if priomap[oneiter.(map[string]interface{})["priority"].(string)] == nil{

							modelmap := make(map[string]interface{})
							kindmap := make(map[string]interface{})
							typemap := make(map[string]interface{})

							onemap := make(map[string]interface{})
							
							onemap["perf"] = oneiter.(map[string]interface{})["value"].(string)
							onemap["competitor"] = oneiter.(map[string]interface{})["competitor"].(string)

							if StringTransferToFloat64(oneiter.(map[string]interface{})["value"].(string)) > 0 && StringTransferToFloat64(oneiter.(map[string]interface{})["competitor"].(string)) > 0{
								onemap["sore"] = fmt.Sprintf("%.4f", (StringTransferToFloat64(oneiter.(map[string]interface{})["value"].(string))/StringTransferToFloat64(oneiter.(map[string]interface{})["competitor"].(string))))
							}else{
								onemap["sore"] = "/"
							}

							if StringTransferToFloat64(onemap["sore"].(string)) > 0 && StringTransferToFloat64(oneiter.(map[string]interface{})["ratio"].(string)) > 0{
								onemap["goalscore"] = fmt.Sprintf("%.4f", (StringTransferToFloat64(onemap["sore"].(string))/StringTransferToFloat64(oneiter.(map[string]interface{})["ratio"].(string))))
							}else{
								onemap["goalscore"] = "/"
							}

							if StringTransferToFloat64(oneiter.(map[string]interface{})["competitor"].(string)) <= 0 && StringTransferToFloat64(onemap["perf"].(string)) > 0{
								onemap["valuetag"] = "noproject"
							}else if StringTransferToFloat64(oneiter.(map[string]interface{})["competitor"].(string)) > 0 && StringTransferToFloat64(onemap["perf"].(string)) <= 0{
								onemap["valuetag"] = "novalue"
							}else if StringTransferToFloat64(oneiter.(map[string]interface{})["competitor"].(string)) > 0 && StringTransferToFloat64(onemap["perf"].(string)) > 0{
								onemap["valuetag"] = "normal"
							}else{
								onemap["value"] = "/"
							}

							// log.Println("onemap", onemap)
							
							
							modelmap[oneiter.(map[string]interface{})["case_name"].(string)] = onemap
							kindmap["Batched inference-2 Tile"] = modelmap
							typemap[oneiter.(map[string]interface{})["precision"].(string)] = kindmap
							priomap[oneiter.(map[string]interface{})["priority"].(string)] = typemap					


						}else{
							// log.Println("来这里了 5555")

							onetypemap := priomap[oneiter.(map[string]interface{})["priority"].(string)]

							if onetypemap.(map[string]interface{})[oneiter.(map[string]interface{})["precision"].(string)] == nil {
								// log.Println("来这里了 6666")
								kindmap2 := make(map[string]interface{})
								modelmap2 := make(map[string]interface{})
								onemap := make(map[string]interface{})

								onemap["perf"] = oneiter.(map[string]interface{})["value"].(string)
								onemap["competitor"] = oneiter.(map[string]interface{})["competitor"].(string)

								if StringTransferToFloat64(oneiter.(map[string]interface{})["value"].(string)) > 0 && StringTransferToFloat64(oneiter.(map[string]interface{})["competitor"].(string)) > 0{
									onemap["sore"] = fmt.Sprintf("%.4f", (StringTransferToFloat64(oneiter.(map[string]interface{})["value"].(string))/StringTransferToFloat64(oneiter.(map[string]interface{})["competitor"].(string))))
								}else{
									onemap["sore"] = "/"
								}

								if StringTransferToFloat64(onemap["sore"].(string)) > 0 && StringTransferToFloat64(oneiter.(map[string]interface{})["ratio"].(string)) > 0{
									onemap["goalscore"] = fmt.Sprintf("%.4f", (StringTransferToFloat64(onemap["sore"].(string))/StringTransferToFloat64(oneiter.(map[string]interface{})["ratio"].(string))))
								}else{
									onemap["goalscore"] = "/"
								}

								if StringTransferToFloat64(oneiter.(map[string]interface{})["competitor"].(string)) <= 0 && StringTransferToFloat64(onemap["perf"].(string)) > 0{
									onemap["valuetag"] = "noproject"
								}else if StringTransferToFloat64(oneiter.(map[string]interface{})["competitor"].(string)) > 0 && StringTransferToFloat64(onemap["perf"].(string)) <= 0{
									onemap["valuetag"] = "novalue"
								}else if StringTransferToFloat64(oneiter.(map[string]interface{})["competitor"].(string)) > 0 && StringTransferToFloat64(onemap["perf"].(string)) > 0{
									onemap["valuetag"] = "normal"
								}else{
									onemap["value"] = "/"
								}

								modelmap2[oneiter.(map[string]interface{})["case_name"].(string)] = onemap
								kindmap2["Batched inference-2 Tile"] = modelmap2
								onetypemap.(map[string]interface{})[oneiter.(map[string]interface{})["precision"].(string)] = kindmap2

							}else{

								// log.Println("来这里了 6666")

								onekindmap := onetypemap.(map[string]interface{})[oneiter.(map[string]interface{})["precision"].(string)]
								onemodelmap := onekindmap.(map[string]interface{})["Batched inference-2 Tile"]

								if onemodelmap != nil{
									if onemodelmap.(map[string]interface{})[oneiter.(map[string]interface{})["case_name"].(string)] == nil{
										// log.Println("444 来过这吗？")
										onemap := make(map[string]interface{})
										onemap["perf"] = oneiter.(map[string]interface{})["value"].(string)
										onemap["competitor"] = oneiter.(map[string]interface{})["competitor"].(string)

										if StringTransferToFloat64(oneiter.(map[string]interface{})["value"].(string)) > 0 && StringTransferToFloat64(oneiter.(map[string]interface{})["competitor"].(string)) > 0{
											onemap["sore"] = fmt.Sprintf("%.4f", (StringTransferToFloat64(oneiter.(map[string]interface{})["value"].(string))/StringTransferToFloat64(oneiter.(map[string]interface{})["competitor"].(string))))
										}else{
											onemap["sore"] = "/"
										}
				
										if StringTransferToFloat64(onemap["sore"].(string)) > 0 && StringTransferToFloat64(oneiter.(map[string]interface{})["ratio"].(string)) > 0{
											onemap["goalscore"] = fmt.Sprintf("%.4f", (StringTransferToFloat64(onemap["sore"].(string))/StringTransferToFloat64(oneiter.(map[string]interface{})["ratio"].(string))))
										}else{
											onemap["goalscore"] = "/"
										}
				
										if StringTransferToFloat64(oneiter.(map[string]interface{})["competitor"].(string)) <= 0 && StringTransferToFloat64(onemap["perf"].(string)) > 0{
											onemap["valuetag"] = "noproject"
										}else if StringTransferToFloat64(oneiter.(map[string]interface{})["competitor"].(string)) > 0 && StringTransferToFloat64(onemap["perf"].(string)) <= 0{
											onemap["valuetag"] = "novalue"
										}else if StringTransferToFloat64(oneiter.(map[string]interface{})["competitor"].(string)) > 0 && StringTransferToFloat64(onemap["perf"].(string)) > 0{
											onemap["valuetag"] = "normal"
										}else{
											onemap["value"] = "/"
										}

										onemodelmap.(map[string]interface{})[oneiter.(map[string]interface{})["case_name"].(string)] = onemap
									}	
								}else{
									onemodelmap = make(map[string]interface{})
									// log.Println("来这里了 7777")
									onemap := make(map[string]interface{})
									onemap["perf"] = oneiter.(map[string]interface{})["value"].(string)
									onemap["competitor"] = oneiter.(map[string]interface{})["competitor"].(string)

									if StringTransferToFloat64(oneiter.(map[string]interface{})["value"].(string)) > 0 && StringTransferToFloat64(oneiter.(map[string]interface{})["competitor"].(string)) > 0{
										onemap["sore"] = fmt.Sprintf("%.4f", (StringTransferToFloat64(oneiter.(map[string]interface{})["value"].(string))/StringTransferToFloat64(oneiter.(map[string]interface{})["competitor"].(string))))
									}else{
										onemap["sore"] = "/"
									}
			
									if StringTransferToFloat64(onemap["sore"].(string)) > 0 && StringTransferToFloat64(oneiter.(map[string]interface{})["ratio"].(string)) > 0{
										onemap["goalscore"] = fmt.Sprintf("%.4f", (StringTransferToFloat64(onemap["sore"].(string))/StringTransferToFloat64(oneiter.(map[string]interface{})["ratio"].(string))))
									}else{
										onemap["goalscore"] = "/"
									}
			
									if StringTransferToFloat64(oneiter.(map[string]interface{})["competitor"].(string)) <= 0 && StringTransferToFloat64(onemap["perf"].(string)) > 0{
										onemap["valuetag"] = "noproject"
									}else if StringTransferToFloat64(oneiter.(map[string]interface{})["competitor"].(string)) > 0 && StringTransferToFloat64(onemap["perf"].(string)) <= 0{
										onemap["valuetag"] = "novalue"
									}else if StringTransferToFloat64(oneiter.(map[string]interface{})["competitor"].(string)) > 0 && StringTransferToFloat64(onemap["perf"].(string)) > 0{
										onemap["valuetag"] = "normal"
									}else{
										onemap["value"] = "/"
									}

									onemodelmap.(map[string]interface{})[oneiter.(map[string]interface{})["case_name"].(string)] = onemap
								}

								
								onekindmap.(map[string]interface{})["Batched inference-2 Tile"] = onemodelmap
								onetypemap.(map[string]interface{})[oneiter.(map[string]interface{})["precision"].(string)] = onekindmap




							}

							priomap[oneiter.(map[string]interface{})["priority"].(string)] = onetypemap


						}




					}

					framemap["Tensorflow"] = priomap
					runmap["inference"] = framemap
					

					for _, oneiter := range por_nt_train_used{

						if runmap["training"] == nil{
							// log.Println("到这里来 888")
							oneframemap := make(map[string]interface{})
							modelmap := make(map[string]interface{})
							kindmap := make(map[string]interface{})
							typemap := make(map[string]interface{})
							onepriomap := make(map[string]interface{})
							onemap := make(map[string]interface{})

							// need a function to find min value for nt
							onemap["perf"] = FindMinValue(oneiter.(map[string]interface{})["log"].([]interface{})) 

							onemap["competitor"] = oneiter.(map[string]interface{})["competitor"].(string)

							if StringTransferToFloat64(oneiter.(map[string]interface{})["value"].(string)) > 0 && StringTransferToFloat64(oneiter.(map[string]interface{})["competitor"].(string)) > 0{
								onemap["sore"] = fmt.Sprintf("%.4f", (StringTransferToFloat64(oneiter.(map[string]interface{})["value"].(string))/StringTransferToFloat64(oneiter.(map[string]interface{})["competitor"].(string))))
							}else{
								onemap["sore"] = "/"
							}

							if StringTransferToFloat64(onemap["sore"].(string)) > 0 && StringTransferToFloat64(oneiter.(map[string]interface{})["ratio"].(string)) > 0{
								onemap["goalscore"] = fmt.Sprintf("%.4f", (StringTransferToFloat64(onemap["sore"].(string))/StringTransferToFloat64(oneiter.(map[string]interface{})["ratio"].(string))))
							}else{
								onemap["goalscore"] = "/"
							}

							if StringTransferToFloat64(oneiter.(map[string]interface{})["competitor"].(string)) <= 0 && StringTransferToFloat64(onemap["perf"].(string)) > 0{
								onemap["valuetag"] = "noproject"
							}else if StringTransferToFloat64(oneiter.(map[string]interface{})["competitor"].(string)) > 0 && StringTransferToFloat64(onemap["perf"].(string)) <= 0{
								onemap["valuetag"] = "novalue"
							}else if StringTransferToFloat64(oneiter.(map[string]interface{})["competitor"].(string)) > 0 && StringTransferToFloat64(onemap["perf"].(string)) > 0{
								onemap["valuetag"] = "normal"
							}else{
								onemap["value"] = "/"
							}
							
							
							modelmap[oneiter.(map[string]interface{})["case_name"].(string)] = onemap
							kindmap["lower throughput in 2 tile run"] = modelmap
							typemap[oneiter.(map[string]interface{})["precision"].(string)] = kindmap
							onepriomap[oneiter.(map[string]interface{})["priority"].(string)] = typemap
							oneframemap["Tensorflow"] = onepriomap
							runmap["training"] = oneframemap
						}else{
							oneframemap := runmap["training"]
							onepriomap := oneframemap.(map[string]interface{})["Tensorflow"]
							// log.Println("到这里来 99999")
							if onepriomap.(map[string]interface{})[oneiter.(map[string]interface{})["priority"].(string)] == nil{
								// log.Println("到这里来 ttttt")
								
								modelmap3 := make(map[string]interface{})
								kindmap3 := make(map[string]interface{})
								typemap3 := make(map[string]interface{})
								
								onemap := make(map[string]interface{})
			
								// need a function to find min value for nt
								onemap["perf"] = FindMinValue(oneiter.(map[string]interface{})["log"].([]interface{})) 
			
								onemap["competitor"] = oneiter.(map[string]interface{})["competitor"].(string)
			
								if StringTransferToFloat64(oneiter.(map[string]interface{})["value"].(string)) > 0 && StringTransferToFloat64(oneiter.(map[string]interface{})["competitor"].(string)) > 0{
									onemap["sore"] = fmt.Sprintf("%.4f", (StringTransferToFloat64(oneiter.(map[string]interface{})["value"].(string))/StringTransferToFloat64(oneiter.(map[string]interface{})["competitor"].(string))))
								}else{
									onemap["sore"] = "/"
								}
			
								if StringTransferToFloat64(onemap["sore"].(string)) > 0 && StringTransferToFloat64(oneiter.(map[string]interface{})["ratio"].(string)) > 0{
									onemap["goalscore"] = fmt.Sprintf("%.4f", (StringTransferToFloat64(onemap["sore"].(string))/StringTransferToFloat64(oneiter.(map[string]interface{})["ratio"].(string))))
								}else{
									onemap["goalscore"] = "/"
								}
			
								if StringTransferToFloat64(oneiter.(map[string]interface{})["competitor"].(string)) <= 0 && StringTransferToFloat64(onemap["perf"].(string)) > 0{
									onemap["valuetag"] = "noproject"
								}else if StringTransferToFloat64(oneiter.(map[string]interface{})["competitor"].(string)) > 0 && StringTransferToFloat64(onemap["perf"].(string)) <= 0{
									onemap["valuetag"] = "novalue"
								}else if StringTransferToFloat64(oneiter.(map[string]interface{})["competitor"].(string)) > 0 && StringTransferToFloat64(onemap["perf"].(string)) > 0{
									onemap["valuetag"] = "normal"
								}else{
									onemap["value"] = "/"
								}
								
								
								modelmap3[oneiter.(map[string]interface{})["case_name"].(string)] = onemap
								kindmap3["lower throughput in 2 tile run"] = modelmap3
								typemap3[oneiter.(map[string]interface{})["precision"].(string)] = kindmap3
								onepriomap.(map[string]interface{})[oneiter.(map[string]interface{})["priority"].(string)] = typemap3

							}else{
								// log.Println("到这里来 xxxxxxx")
								onetypemap := onepriomap.(map[string]interface{})[oneiter.(map[string]interface{})["priority"].(string)]

								if onetypemap.(map[string]interface{})[oneiter.(map[string]interface{})["precision"].(string)] == nil {
									// log.Println("到这里来 ccccccccccccc")

									kindmap4 := make(map[string]interface{})
									modelmap4 := make(map[string]interface{})
									onemap := make(map[string]interface{})

									onemap["perf"] = FindMinValue(oneiter.(map[string]interface{})["log"].([]interface{})) 
			
									onemap["competitor"] = oneiter.(map[string]interface{})["competitor"].(string)
				
									if StringTransferToFloat64(oneiter.(map[string]interface{})["value"].(string)) > 0 && StringTransferToFloat64(oneiter.(map[string]interface{})["competitor"].(string)) > 0{
										onemap["sore"] = fmt.Sprintf("%.4f", (StringTransferToFloat64(oneiter.(map[string]interface{})["value"].(string))/StringTransferToFloat64(oneiter.(map[string]interface{})["competitor"].(string))))
									}else{
										onemap["sore"] = "/"
									}
				
									if StringTransferToFloat64(onemap["sore"].(string)) > 0 && StringTransferToFloat64(oneiter.(map[string]interface{})["ratio"].(string)) > 0{
										onemap["goalscore"] = fmt.Sprintf("%.4f", (StringTransferToFloat64(onemap["sore"].(string))/StringTransferToFloat64(oneiter.(map[string]interface{})["ratio"].(string))))
									}else{
										onemap["goalscore"] = "/"
									}
				
									if StringTransferToFloat64(oneiter.(map[string]interface{})["competitor"].(string)) <= 0 && StringTransferToFloat64(onemap["perf"].(string)) > 0{
										onemap["valuetag"] = "noproject"
									}else if StringTransferToFloat64(oneiter.(map[string]interface{})["competitor"].(string)) > 0 && StringTransferToFloat64(onemap["perf"].(string)) <= 0{
										onemap["valuetag"] = "novalue"
									}else if StringTransferToFloat64(oneiter.(map[string]interface{})["competitor"].(string)) > 0 && StringTransferToFloat64(onemap["perf"].(string)) > 0{
										onemap["valuetag"] = "normal"
									}else{
										onemap["value"] = "/"
									}

									modelmap4[oneiter.(map[string]interface{})["case_name"].(string)] = onemap
									kindmap4["lower throughput in 2 tile run"] = modelmap4
									onetypemap.(map[string]interface{})[oneiter.(map[string]interface{})["precision"].(string)] = kindmap4

								}else{

									// log.Println("到这里来 dddddd")

									onekindmap := onetypemap.(map[string]interface{})[oneiter.(map[string]interface{})["precision"].(string)]
									onemodelmap := onekindmap.(map[string]interface{})["lower throughput in 2 tile run"]

									if onemodelmap != nil{
										if onemodelmap.(map[string]interface{})[oneiter.(map[string]interface{})["case_name"].(string)] == nil{
											// log.Println("到这里来 hhhhhhhh")

											onemap := make(map[string]interface{})
											onemap["perf"] = FindMinValue(oneiter.(map[string]interface{})["log"].([]interface{})) 
			
											onemap["competitor"] = oneiter.(map[string]interface{})["competitor"].(string)
						
											if StringTransferToFloat64(oneiter.(map[string]interface{})["value"].(string)) > 0 && StringTransferToFloat64(oneiter.(map[string]interface{})["competitor"].(string)) > 0{
												onemap["sore"] = fmt.Sprintf("%.4f", (StringTransferToFloat64(oneiter.(map[string]interface{})["value"].(string))/StringTransferToFloat64(oneiter.(map[string]interface{})["competitor"].(string))))
											}else{
												onemap["sore"] = "/"
											}
						
											if StringTransferToFloat64(onemap["sore"].(string)) > 0 && StringTransferToFloat64(oneiter.(map[string]interface{})["ratio"].(string)) > 0{
												onemap["goalscore"] = fmt.Sprintf("%.4f", (StringTransferToFloat64(onemap["sore"].(string))/StringTransferToFloat64(oneiter.(map[string]interface{})["ratio"].(string))))
											}else{
												onemap["goalscore"] = "/"
											}
						
											if StringTransferToFloat64(oneiter.(map[string]interface{})["competitor"].(string)) <= 0 && StringTransferToFloat64(onemap["perf"].(string)) > 0{
												onemap["valuetag"] = "noproject"
											}else if StringTransferToFloat64(oneiter.(map[string]interface{})["competitor"].(string)) > 0 && StringTransferToFloat64(onemap["perf"].(string)) <= 0{
												onemap["valuetag"] = "novalue"
											}else if StringTransferToFloat64(oneiter.(map[string]interface{})["competitor"].(string)) > 0 && StringTransferToFloat64(onemap["perf"].(string)) > 0{
												onemap["valuetag"] = "normal"
											}else{
												onemap["value"] = "/"
											}

											onemodelmap.(map[string]interface{})[oneiter.(map[string]interface{})["case_name"].(string)] = onemap

										}


									}else{

										onemodelmap = make(map[string]interface{})
										// log.Println("到这里来 sssssss")

										onemap := make(map[string]interface{})
										onemap["perf"] = FindMinValue(oneiter.(map[string]interface{})["log"].([]interface{})) 

										onemap["competitor"] = oneiter.(map[string]interface{})["competitor"].(string)
					
										if StringTransferToFloat64(oneiter.(map[string]interface{})["value"].(string)) > 0 && StringTransferToFloat64(oneiter.(map[string]interface{})["competitor"].(string)) > 0{
											onemap["sore"] = fmt.Sprintf("%.4f", (StringTransferToFloat64(oneiter.(map[string]interface{})["value"].(string))/StringTransferToFloat64(oneiter.(map[string]interface{})["competitor"].(string))))
										}else{
											onemap["sore"] = "/"
										}
					
										if StringTransferToFloat64(onemap["sore"].(string)) > 0 && StringTransferToFloat64(oneiter.(map[string]interface{})["ratio"].(string)) > 0{
											onemap["goalscore"] = fmt.Sprintf("%.4f", (StringTransferToFloat64(onemap["sore"].(string))/StringTransferToFloat64(oneiter.(map[string]interface{})["ratio"].(string))))
										}else{
											onemap["goalscore"] = "/"
										}
					
										if StringTransferToFloat64(oneiter.(map[string]interface{})["competitor"].(string)) <= 0 && StringTransferToFloat64(onemap["perf"].(string)) > 0{
											onemap["valuetag"] = "noproject"
										}else if StringTransferToFloat64(oneiter.(map[string]interface{})["competitor"].(string)) > 0 && StringTransferToFloat64(onemap["perf"].(string)) <= 0{
											onemap["valuetag"] = "novalue"
										}else if StringTransferToFloat64(oneiter.(map[string]interface{})["competitor"].(string)) > 0 && StringTransferToFloat64(onemap["perf"].(string)) > 0{
											onemap["valuetag"] = "normal"
										}else{
											onemap["value"] = "/"
										}

										onemodelmap.(map[string]interface{})[oneiter.(map[string]interface{})["case_name"].(string)] = onemap




									}


									onekindmap.(map[string]interface{})["lower throughput in 2 tile run"] = onemodelmap
									onetypemap.(map[string]interface{})[oneiter.(map[string]interface{})["precision"].(string)] = onekindmap


								}

								onepriomap.(map[string]interface{})[oneiter.(map[string]interface{})["priority"].(string)] = onetypemap
							}


							oneframemap.(map[string]interface{})["Tensorflow"] = onepriomap
							runmap["training"] = oneframemap







						}


					}


					for _, oneiter := range por_hvd_used{
						
						if runmap["training"] == nil{
							// log.Println("快到碗里来 1111")

							oneframemap := make(map[string]interface{})
							modelmap := make(map[string]interface{})
							kindmap := make(map[string]interface{})
							typemap := make(map[string]interface{})
							onepriomap := make(map[string]interface{})

							onemap := make(map[string]interface{})
									
							onemap["perf"] = oneiter.(map[string]interface{})["value"].(string)
							onemap["competitor"] = oneiter.(map[string]interface{})["competitor"].(string)

							if StringTransferToFloat64(oneiter.(map[string]interface{})["value"].(string)) > 0 && StringTransferToFloat64(oneiter.(map[string]interface{})["competitor"].(string)) > 0{
								onemap["sore"] = fmt.Sprintf("%.4f", (StringTransferToFloat64(oneiter.(map[string]interface{})["value"].(string))/StringTransferToFloat64(oneiter.(map[string]interface{})["competitor"].(string))))
							}else{
								onemap["sore"] = "/"
							}

							if StringTransferToFloat64(onemap["sore"].(string)) > 0 && StringTransferToFloat64(oneiter.(map[string]interface{})["ratio"].(string)) > 0{
								onemap["goalscore"] = fmt.Sprintf("%.4f", (StringTransferToFloat64(onemap["sore"].(string))/StringTransferToFloat64(oneiter.(map[string]interface{})["ratio"].(string))))
							}else{
								onemap["goalscore"] = "/"
							}

							if StringTransferToFloat64(oneiter.(map[string]interface{})["competitor"].(string)) <= 0 && StringTransferToFloat64(onemap["perf"].(string)) > 0{
								onemap["valuetag"] = "noproject"
							}else if StringTransferToFloat64(oneiter.(map[string]interface{})["competitor"].(string)) > 0 && StringTransferToFloat64(onemap["perf"].(string)) <= 0{
								onemap["valuetag"] = "novalue"
							}else if StringTransferToFloat64(oneiter.(map[string]interface{})["competitor"].(string)) > 0 && StringTransferToFloat64(onemap["perf"].(string)) > 0{
								onemap["valuetag"] = "normal"
							}else{
								onemap["value"] = "/"
							}
							
							modelmap[oneiter.(map[string]interface{})["case_name"].(string)] = onemap
							kindmap["2T distributed training"] = modelmap
							typemap[oneiter.(map[string]interface{})["precision"].(string)] = kindmap
							onepriomap[oneiter.(map[string]interface{})["priority"].(string)] = typemap
							oneframemap["Tensorflow"] = onepriomap
							runmap["training"] = oneframemap

						}else{
							// log.Println("快到碗里来 2222")

							oneframemap := runmap["training"]
							onepriomap := oneframemap.(map[string]interface{})["Tensorflow"]

							if onepriomap.(map[string]interface{})[oneiter.(map[string]interface{})["priority"].(string)] == nil{
								// log.Println("快到碗里来 3333")

								modelmap6 := make(map[string]interface{})
								kindmap6 := make(map[string]interface{})
								typemap6 := make(map[string]interface{})

								onemap := make(map[string]interface{})
									
								onemap["perf"] = oneiter.(map[string]interface{})["value"].(string)
								onemap["competitor"] = oneiter.(map[string]interface{})["competitor"].(string)
			
								if StringTransferToFloat64(oneiter.(map[string]interface{})["value"].(string)) > 0 && StringTransferToFloat64(oneiter.(map[string]interface{})["competitor"].(string)) > 0{
									onemap["sore"] = fmt.Sprintf("%.4f", (StringTransferToFloat64(oneiter.(map[string]interface{})["value"].(string))/StringTransferToFloat64(oneiter.(map[string]interface{})["competitor"].(string))))
								}else{
									onemap["sore"] = "/"
								}
			
								if StringTransferToFloat64(onemap["sore"].(string)) > 0 && StringTransferToFloat64(oneiter.(map[string]interface{})["ratio"].(string)) > 0{
									onemap["goalscore"] = fmt.Sprintf("%.4f", (StringTransferToFloat64(onemap["sore"].(string))/StringTransferToFloat64(oneiter.(map[string]interface{})["ratio"].(string))))
								}else{
									onemap["goalscore"] = "/"
								}
			
								if StringTransferToFloat64(oneiter.(map[string]interface{})["competitor"].(string)) <= 0 && StringTransferToFloat64(onemap["perf"].(string)) > 0{
									onemap["valuetag"] = "noproject"
								}else if StringTransferToFloat64(oneiter.(map[string]interface{})["competitor"].(string)) > 0 && StringTransferToFloat64(onemap["perf"].(string)) <= 0{
									onemap["valuetag"] = "novalue"
								}else if StringTransferToFloat64(oneiter.(map[string]interface{})["competitor"].(string)) > 0 && StringTransferToFloat64(onemap["perf"].(string)) > 0{
									onemap["valuetag"] = "normal"
								}else{
									onemap["value"] = "/"
								}
								
								modelmap6[oneiter.(map[string]interface{})["case_name"].(string)] = onemap
								kindmap6["2T distributed training"] = modelmap6
								typemap6[oneiter.(map[string]interface{})["precision"].(string)] = kindmap6
								onepriomap.(map[string]interface{})[oneiter.(map[string]interface{})["priority"].(string)] = typemap6


							}else{

								// log.Println("快到碗里来 4444")
								onetypemap := onepriomap.(map[string]interface{})[oneiter.(map[string]interface{})["priority"].(string)]

								if onetypemap.(map[string]interface{})[oneiter.(map[string]interface{})["precision"].(string)] == nil {

									// log.Println("快到碗里来 55555")

									modelmap7 := make(map[string]interface{})
									kindmap7 := make(map[string]interface{})
								
									onemap := make(map[string]interface{})
									
									onemap["perf"] = oneiter.(map[string]interface{})["value"].(string)
									onemap["competitor"] = oneiter.(map[string]interface{})["competitor"].(string)
				
									if StringTransferToFloat64(oneiter.(map[string]interface{})["value"].(string)) > 0 && StringTransferToFloat64(oneiter.(map[string]interface{})["competitor"].(string)) > 0{
										onemap["sore"] = fmt.Sprintf("%.4f", (StringTransferToFloat64(oneiter.(map[string]interface{})["value"].(string))/StringTransferToFloat64(oneiter.(map[string]interface{})["competitor"].(string))))
									}else{
										onemap["sore"] = "/"
									}
				
									if StringTransferToFloat64(onemap["sore"].(string)) > 0 && StringTransferToFloat64(oneiter.(map[string]interface{})["ratio"].(string)) > 0{
										onemap["goalscore"] = fmt.Sprintf("%.4f", (StringTransferToFloat64(onemap["sore"].(string))/StringTransferToFloat64(oneiter.(map[string]interface{})["ratio"].(string))))
									}else{
										onemap["goalscore"] = "/"
									}
				
									if StringTransferToFloat64(oneiter.(map[string]interface{})["competitor"].(string)) <= 0 && StringTransferToFloat64(onemap["perf"].(string)) > 0{
										onemap["valuetag"] = "noproject"
									}else if StringTransferToFloat64(oneiter.(map[string]interface{})["competitor"].(string)) > 0 && StringTransferToFloat64(onemap["perf"].(string)) <= 0{
										onemap["valuetag"] = "novalue"
									}else if StringTransferToFloat64(oneiter.(map[string]interface{})["competitor"].(string)) > 0 && StringTransferToFloat64(onemap["perf"].(string)) > 0{
										onemap["valuetag"] = "normal"
									}else{
										onemap["value"] = "/"
									}

									modelmap7[oneiter.(map[string]interface{})["case_name"].(string)] = onemap
									kindmap7["2T distributed training"] = modelmap7
									onetypemap.(map[string]interface{})[oneiter.(map[string]interface{})["precision"].(string)] = kindmap7





								}else{
									// log.Println("快到碗里来 6666")

									onekindmap := onetypemap.(map[string]interface{})[oneiter.(map[string]interface{})["precision"].(string)]
									onemodelmap := onekindmap.(map[string]interface{})["2T distributed training"]

									if onemodelmap != nil{
										if onemodelmap.(map[string]interface{})[oneiter.(map[string]interface{})["case_name"].(string)] == nil{

											// log.Println("快到碗里来 7777")

											onemap := make(map[string]interface{})
									
											onemap["perf"] = oneiter.(map[string]interface{})["value"].(string)
											onemap["competitor"] = oneiter.(map[string]interface{})["competitor"].(string)
						
											if StringTransferToFloat64(oneiter.(map[string]interface{})["value"].(string)) > 0 && StringTransferToFloat64(oneiter.(map[string]interface{})["competitor"].(string)) > 0{
												onemap["sore"] = fmt.Sprintf("%.4f", (StringTransferToFloat64(oneiter.(map[string]interface{})["value"].(string))/StringTransferToFloat64(oneiter.(map[string]interface{})["competitor"].(string))))
											}else{
												onemap["sore"] = "/"
											}
						
											if StringTransferToFloat64(onemap["sore"].(string)) > 0 && StringTransferToFloat64(oneiter.(map[string]interface{})["ratio"].(string)) > 0{
												onemap["goalscore"] = fmt.Sprintf("%.4f", (StringTransferToFloat64(onemap["sore"].(string))/StringTransferToFloat64(oneiter.(map[string]interface{})["ratio"].(string))))
											}else{
												onemap["goalscore"] = "/"
											}
						
											if StringTransferToFloat64(oneiter.(map[string]interface{})["competitor"].(string)) <= 0 && StringTransferToFloat64(onemap["perf"].(string)) > 0{
												onemap["valuetag"] = "noproject"
											}else if StringTransferToFloat64(oneiter.(map[string]interface{})["competitor"].(string)) > 0 && StringTransferToFloat64(onemap["perf"].(string)) <= 0{
												onemap["valuetag"] = "novalue"
											}else if StringTransferToFloat64(oneiter.(map[string]interface{})["competitor"].(string)) > 0 && StringTransferToFloat64(onemap["perf"].(string)) > 0{
												onemap["valuetag"] = "normal"
											}else{
												onemap["value"] = "/"
											}

											onemodelmap.(map[string]interface{})[oneiter.(map[string]interface{})["case_name"].(string)] = onemap



										}


									}else{

										// log.Println("快到碗里来 8888")

										onemodelmap = make(map[string]interface{})

										onemap := make(map[string]interface{})
									
										onemap["perf"] = oneiter.(map[string]interface{})["value"].(string)
										onemap["competitor"] = oneiter.(map[string]interface{})["competitor"].(string)
					
										if StringTransferToFloat64(oneiter.(map[string]interface{})["value"].(string)) > 0 && StringTransferToFloat64(oneiter.(map[string]interface{})["competitor"].(string)) > 0{
											onemap["sore"] = fmt.Sprintf("%.4f", (StringTransferToFloat64(oneiter.(map[string]interface{})["value"].(string))/StringTransferToFloat64(oneiter.(map[string]interface{})["competitor"].(string))))
										}else{
											onemap["sore"] = "/"
										}
					
										if StringTransferToFloat64(onemap["sore"].(string)) > 0 && StringTransferToFloat64(oneiter.(map[string]interface{})["ratio"].(string)) > 0{
											onemap["goalscore"] = fmt.Sprintf("%.4f", (StringTransferToFloat64(onemap["sore"].(string))/StringTransferToFloat64(oneiter.(map[string]interface{})["ratio"].(string))))
										}else{
											onemap["goalscore"] = "/"
										}
					
										if StringTransferToFloat64(oneiter.(map[string]interface{})["competitor"].(string)) <= 0 && StringTransferToFloat64(onemap["perf"].(string)) > 0{
											onemap["valuetag"] = "noproject"
										}else if StringTransferToFloat64(oneiter.(map[string]interface{})["competitor"].(string)) > 0 && StringTransferToFloat64(onemap["perf"].(string)) <= 0{
											onemap["valuetag"] = "novalue"
										}else if StringTransferToFloat64(oneiter.(map[string]interface{})["competitor"].(string)) > 0 && StringTransferToFloat64(onemap["perf"].(string)) > 0{
											onemap["valuetag"] = "normal"
										}else{
											onemap["value"] = "/"
										}

										onemodelmap.(map[string]interface{})[oneiter.(map[string]interface{})["case_name"].(string)] = onemap


									}

									onekindmap.(map[string]interface{})["2T distributed training"] = onemodelmap
									onetypemap.(map[string]interface{})[oneiter.(map[string]interface{})["precision"].(string)] = onekindmap


								}

								onepriomap.(map[string]interface{})[oneiter.(map[string]interface{})["priority"].(string)] = onetypemap


							}

							oneframemap.(map[string]interface{})["Tensorflow"] = onepriomap
							runmap["training"] = oneframemap
						}




					}





					resultmap = runmap


				}

			
			}else{
				log.Println("Sorry, no data")
			}	
			
		case "ATSM":
			if errpipr == nil && len(dateinterface) > 0{

				por_1t_infer := make([]interface{},0)
				por_1t_infer_used := make([]interface{},0)

				for _, oneGe := range dataSet["general"].([]interface{}){
					if strings.ToLower(strings.Split(oneGe.(map[string]interface{})["r2_type"].(string), "@")[0]) == "inference"{
						log.Println("oneGe keyname", oneGe.(map[string]interface{})["key_name"].(string))
						por_1t_infer = oneGe.(map[string]interface{})["data"].([]interface{})
					}
				}

				for _, onetime := range dateinterface{

					if onetime.(bson.M)["time"].(int64) > newtime {
						newtime = onetime.(bson.M)["time"].(int64)  //latest
					}
				}
				log.Println("you & me", newtime)
				competitorbson := make(map[string]interface{})
				queryBson := bson.M{"time":newtime, "hardware":hardware}
				err := c.Find(queryBson).One(&competitorbson)

				if err == nil{
					plist := make([]string, 0)
					typelist := make([]string, 0)
					modellist := make([]string, 0)


					priomap := make(map[string]interface{})
					framemap := make(map[string]interface{})
					runmap := make(map[string]interface{})

					if len(por_1t_infer) > 0{
						log.Println("len por_1t_infer", len(por_1t_infer))
						for _, oneData := range por_1t_infer{
							keyname := oneData.(map[string]interface{})["test_mode"].(string) + " " + oneData.(map[string]interface{})["case_name"].(string) + " " + 
								oneData.(map[string]interface{})["layout"].(string) + " " + oneData.(map[string]interface{})["precision"].(string) + " " + 
								oneData.(map[string]interface{})["run_type"].(string) + " " + oneData.(map[string]interface{})["batch_size"].(string) + " " + 
								oneData.(map[string]interface{})["scenario"].(string)
								log.Println("you:", keyname)

							for _, onecom := range competitorbson["competitor"].([]interface{}){
								
								
								if keyname == onecom.(map[string]interface{})["key_name"] && onecom.(map[string]interface{})["priority"] != "N/A"{

									oneData.(map[string]interface{})["priority"] = strings.ToUpper(onecom.(map[string]interface{})["priority"].(string))
									oneData.(map[string]interface{})["competitor"] = onecom.(map[string]interface{})["competitor_number"]
									oneData.(map[string]interface{})["ratio"] = onecom.(map[string]interface{})["pri-multiplier"]
		
									por_1t_infer_used = append(por_1t_infer_used, oneData)
								}
							}
						}

					}

					log.Println("por_1t_infer_used", len(por_1t_infer_used))
					for _, oneiter := range por_1t_infer_used{
				
						if checkDuplicateString(plist, oneiter.(map[string]interface{})["priority"].(string)){
							log.Println("PPPP",oneiter.(map[string]interface{})["priority"].(string))
							plist = append(plist, oneiter.(map[string]interface{})["priority"].(string))
							if checkDuplicateString(typelist, oneiter.(map[string]interface{})["precision"].(string)){
								typelist = append(typelist, oneiter.(map[string]interface{})["precision"].(string))
								
								if checkDuplicateString(modellist, oneiter.(map[string]interface{})["case_name"].(string)){
		
									modellist = append(modellist, oneiter.(map[string]interface{})["case_name"].(string))
		
									modelmap := make(map[string]interface{})
									kindmap := make(map[string]interface{})
									typemap := make(map[string]interface{})
									onemap := make(map[string]interface{})
									
									onemap["perf"] = oneiter.(map[string]interface{})["value"].(string)
									onemap["competitor"] = oneiter.(map[string]interface{})["competitor"].(string)
		
									if StringTransferToFloat64(oneiter.(map[string]interface{})["value"].(string)) > 0 && StringTransferToFloat64(oneiter.(map[string]interface{})["competitor"].(string)) > 0{
										onemap["sore"] = fmt.Sprintf("%.4f", (StringTransferToFloat64(oneiter.(map[string]interface{})["value"].(string))/StringTransferToFloat64(oneiter.(map[string]interface{})["competitor"].(string))))
									}else{
										onemap["sore"] = "/"
									}
			
									if StringTransferToFloat64(onemap["sore"].(string)) > 0 && StringTransferToFloat64(oneiter.(map[string]interface{})["ratio"].(string)) > 0{
										onemap["goalscore"] = fmt.Sprintf("%.4f", (StringTransferToFloat64(onemap["sore"].(string))/StringTransferToFloat64(oneiter.(map[string]interface{})["ratio"].(string))))
									}else{
										onemap["goalscore"] = "/"
									}
			
									if StringTransferToFloat64(oneiter.(map[string]interface{})["competitor"].(string)) <= 0 && StringTransferToFloat64(onemap["perf"].(string)) > 0{
										onemap["valuetag"] = "noproject"
									}else if StringTransferToFloat64(oneiter.(map[string]interface{})["competitor"].(string)) > 0 && StringTransferToFloat64(onemap["perf"].(string)) <= 0{
										onemap["valuetag"] = "novalue"
									}else if StringTransferToFloat64(oneiter.(map[string]interface{})["competitor"].(string)) > 0 && StringTransferToFloat64(onemap["perf"].(string)) > 0{
										onemap["valuetag"] = "normal"
									}else{
										onemap["value"] = "/"
									}
									
									
									modelmap[oneiter.(map[string]interface{})["case_name"].(string)] = onemap
									kindmap["Batched inference"] = modelmap
									typemap[oneiter.(map[string]interface{})["precision"].(string)] = kindmap
									priomap[oneiter.(map[string]interface{})["priority"].(string)] = typemap
		
								}
		
								
								
								
								
							}else{
								log.Println("来过这吗？")
								log.Println("priomap", priomap)
								onetypemap := priomap[oneiter.(map[string]interface{})["priority"].(string)]
								onekindmap := onetypemap.(map[string]interface{})[oneiter.(map[string]interface{})["precision"].(string)]
								onemodelmap := onekindmap.(map[string]interface{})["Batched inference"]
		
								if onemodelmap.(map[string]interface{})[oneiter.(map[string]interface{})["case_name"].(string)] == nil{
									onemap := make(map[string]interface{})
									onemap["perf"] = oneiter.(map[string]interface{})["value"].(string)
									onemap["competitor"] = oneiter.(map[string]interface{})["competitor"].(string)
		
									if StringTransferToFloat64(oneiter.(map[string]interface{})["value"].(string)) > 0 && StringTransferToFloat64(oneiter.(map[string]interface{})["competitor"].(string)) > 0{
										onemap["sore"] = fmt.Sprintf("%.4f", (StringTransferToFloat64(oneiter.(map[string]interface{})["value"].(string))/StringTransferToFloat64(oneiter.(map[string]interface{})["competitor"].(string))))
									}else{
										onemap["sore"] = "/"
									}
			
									if StringTransferToFloat64(onemap["sore"].(string)) > 0 && StringTransferToFloat64(oneiter.(map[string]interface{})["ratio"].(string)) > 0{
										onemap["goalscore"] = fmt.Sprintf("%.4f", (StringTransferToFloat64(onemap["sore"].(string))/StringTransferToFloat64(oneiter.(map[string]interface{})["ratio"].(string))))
									}else{
										onemap["goalscore"] = "/"
									}
			
									if StringTransferToFloat64(oneiter.(map[string]interface{})["competitor"].(string)) <= 0 && StringTransferToFloat64(onemap["perf"].(string)) > 0{
										onemap["valuetag"] = "noproject"
									}else if StringTransferToFloat64(oneiter.(map[string]interface{})["competitor"].(string)) > 0 && StringTransferToFloat64(onemap["perf"].(string)) <= 0{
										onemap["valuetag"] = "novalue"
									}else if StringTransferToFloat64(oneiter.(map[string]interface{})["competitor"].(string)) > 0 && StringTransferToFloat64(onemap["perf"].(string)) > 0{
										onemap["valuetag"] = "normal"
									}else{
										onemap["value"] = "/"
									}
		
									onemodelmap.(map[string]interface{})[oneiter.(map[string]interface{})["case_name"].(string)] = onemap
								}
								onekindmap.(map[string]interface{})["Batched inference"] = onemodelmap
								onetypemap.(map[string]interface{})[oneiter.(map[string]interface{})["precision"].(string)] = onekindmap
								priomap[oneiter.(map[string]interface{})["priority"].(string)] = onetypemap
		
							}
		
		
						}else{
		
							log.Println("111 来过这吗？")
							log.Println("priomap mama", priomap)
							onetypemap := priomap[oneiter.(map[string]interface{})["priority"].(string)]
							if onetypemap.(map[string]interface{})[oneiter.(map[string]interface{})["precision"].(string)] == nil {
								log.Println("222 来过这吗？")
								kindmap2 := make(map[string]interface{})
								modelmap2 := make(map[string]interface{})
								onemap := make(map[string]interface{})
								onemap["perf"] = oneiter.(map[string]interface{})["value"].(string)
								onemap["competitor"] = oneiter.(map[string]interface{})["competitor"].(string)
		
								if StringTransferToFloat64(oneiter.(map[string]interface{})["value"].(string)) > 0 && StringTransferToFloat64(oneiter.(map[string]interface{})["competitor"].(string)) > 0{
									onemap["sore"] = fmt.Sprintf("%.4f", (StringTransferToFloat64(oneiter.(map[string]interface{})["value"].(string))/StringTransferToFloat64(oneiter.(map[string]interface{})["competitor"].(string))))
								}else{
									onemap["sore"] = "/"
								}
		
								if StringTransferToFloat64(onemap["sore"].(string)) > 0 && StringTransferToFloat64(oneiter.(map[string]interface{})["ratio"].(string)) > 0{
									onemap["goalscore"] = fmt.Sprintf("%.4f", (StringTransferToFloat64(onemap["sore"].(string))/StringTransferToFloat64(oneiter.(map[string]interface{})["ratio"].(string))))
								}else{
									onemap["goalscore"] = "/"
								}
		
								if StringTransferToFloat64(oneiter.(map[string]interface{})["competitor"].(string)) <= 0 && StringTransferToFloat64(onemap["perf"].(string)) > 0{
									onemap["valuetag"] = "noproject"
								}else if StringTransferToFloat64(oneiter.(map[string]interface{})["competitor"].(string)) > 0 && StringTransferToFloat64(onemap["perf"].(string)) <= 0{
									onemap["valuetag"] = "novalue"
								}else if StringTransferToFloat64(oneiter.(map[string]interface{})["competitor"].(string)) > 0 && StringTransferToFloat64(onemap["perf"].(string)) > 0{
									onemap["valuetag"] = "normal"
								}else{
									onemap["value"] = "/"
								}
		
								
								modelmap2[oneiter.(map[string]interface{})["case_name"].(string)] = onemap
								kindmap2["Batched inference"] = modelmap2
								onetypemap.(map[string]interface{})[oneiter.(map[string]interface{})["precision"].(string)] = kindmap2
								
							}else{
								log.Println("333 来过这吗？")
		
								onekindmap := onetypemap.(map[string]interface{})[oneiter.(map[string]interface{})["precision"].(string)]
								onemodelmap := onekindmap.(map[string]interface{})["Batched inference"]
		
								if onemodelmap.(map[string]interface{})[oneiter.(map[string]interface{})["case_name"].(string)] == nil{
									log.Println("444 来过这吗？")
									onemap := make(map[string]interface{})
									onemap["perf"] = oneiter.(map[string]interface{})["value"].(string)
									onemap["competitor"] = oneiter.(map[string]interface{})["competitor"].(string)
		
									if StringTransferToFloat64(oneiter.(map[string]interface{})["value"].(string)) > 0 && StringTransferToFloat64(oneiter.(map[string]interface{})["competitor"].(string)) > 0{
										onemap["sore"] = fmt.Sprintf("%.4f", (StringTransferToFloat64(oneiter.(map[string]interface{})["value"].(string))/StringTransferToFloat64(oneiter.(map[string]interface{})["competitor"].(string))))
									}else{
										onemap["sore"] = "/"
									}
			
									if StringTransferToFloat64(onemap["sore"].(string)) > 0 && StringTransferToFloat64(oneiter.(map[string]interface{})["ratio"].(string)) > 0{
										onemap["goalscore"] = fmt.Sprintf("%.4f", (StringTransferToFloat64(onemap["sore"].(string))/StringTransferToFloat64(oneiter.(map[string]interface{})["ratio"].(string))))
									}else{
										onemap["goalscore"] = "/"
									}
			
									if StringTransferToFloat64(oneiter.(map[string]interface{})["competitor"].(string)) <= 0 && StringTransferToFloat64(onemap["perf"].(string)) > 0{
										onemap["valuetag"] = "noproject"
									}else if StringTransferToFloat64(oneiter.(map[string]interface{})["competitor"].(string)) > 0 && StringTransferToFloat64(onemap["perf"].(string)) <= 0{
										onemap["valuetag"] = "novalue"
									}else if StringTransferToFloat64(oneiter.(map[string]interface{})["competitor"].(string)) > 0 && StringTransferToFloat64(onemap["perf"].(string)) > 0{
										onemap["valuetag"] = "normal"
									}else{
										onemap["value"] = "/"
									}
		
									onemodelmap.(map[string]interface{})[oneiter.(map[string]interface{})["case_name"].(string)] = onemap
								}
								onekindmap.(map[string]interface{})["Batched inference"] = onemodelmap
								onetypemap.(map[string]interface{})[oneiter.(map[string]interface{})["precision"].(string)] = onekindmap
								
							}
		
		
							priomap[oneiter.(map[string]interface{})["priority"].(string)] = onetypemap
		
		
						}
					}

					framemap["Tensorflow"] = priomap
					runmap["inference"] = framemap

					resultmap = runmap
				}




			}


		
	}










	return resultmap



}


func FindMinValue(valuelist []interface{})string{
	minvalue := "100000"
	for _, oneiter := range valuelist{
		for _, onevalue := range oneiter.(map[string]interface{}){
			if StringTransferToFloat64(minvalue) > StringTransferToFloat64(onevalue.([]interface {})[0].(string)){
				minvalue = onevalue.([]interface {})[0].(string)
			}
		}
	}
	return minvalue

}