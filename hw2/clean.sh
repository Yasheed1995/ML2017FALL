#!/usr/bin/env bash 

if osascript -e 'tell app "System Events" to display dialog "Do you want to clean?"'; then
	echo "cleaning..."
	rm -rf generative_output
	rm -rf generative_params
	rm -rf logistic_output
	rm -rf logistic_params
	rm -rf best_out
	if osascript -e 'tell app "System Events" to display dialog "Clean features?"'; then
		rm -rf feature
	fi
else
	echo "quit!"
fi