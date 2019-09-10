/*
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
*/

import React from 'react';
import axios from 'axios';
import colormap from 'colormap';

import {range, flatten} from 'lodash';
import godash from 'godash';
import {Goban} from 'react-go-board';

import Button from '@material-ui/core/Button';
import ToggleButtonGroup from '@material-ui/lab/ToggleButtonGroup';
import ToggleButton from '@material-ui/lab/ToggleButton';
import Container from '@material-ui/core/Container';
import Grid from '@material-ui/core/Grid';
import AppBar from '@material-ui/core/AppBar';
import Toolbar from '@material-ui/core/Toolbar';
import Typography from '@material-ui/core/Typography';
import Table from '@material-ui/core/Table';
import TableBody from '@material-ui/core/TableBody';
import TableCell from '@material-ui/core/TableCell';
import TableHead from '@material-ui/core/TableHead';
import TableSortLabel from '@material-ui/core/TableSortLabel';
import TableRow from '@material-ui/core/TableRow';
import Paper from '@material-ui/core/Paper';
import {styled, ThemeProvider} from '@material-ui/styles';
import {createMuiTheme} from '@material-ui/core/styles';
import IconButton from '@material-ui/core/IconButton';
import KeyboardArrowLeft from '@material-ui/icons/KeyboardArrowLeft';
import KeyboardArrowRight from '@material-ui/icons/KeyboardArrowRight';

import Chart from 'react-google-charts';
import KeyboardEventHandler from 'react-keyboard-event-handler';

import './App.css';


const NUM_COLORS = 48
const MyButton = styled(Button)({
    margin: 10,
});

const colors = colormap({
  colormap: 'copper',
  nshades: NUM_COLORS+1,
  format: 'rgbaString',
  alpha: [0.02,1]
});

const theme = createMuiTheme({
  palette: {
    primary: { main: '#1976d2' },
    secondary: { main: '#D32F2F' }
  }
});


// Lol "const"
const defaultHighlights = {}
defaultHighlights[colors[5]] = flatten(range(10).map(idx => {
  return range(10-idx).map(jdx => {
    return {x: 18-idx, y:9-jdx };
  });
}));

class DefaultDict {
  constructor(defaultInit) {
    return new Proxy({}, {
      get: (target, name) => name in target ?
        target[name] :
        (target[name] = typeof defaultInit === 'function' ?
          new defaultInit().valueOf() :
          defaultInit)
      })
  }
}

const topLeft = {x: 9, y: 0};
const bottomRight = {x: 18, y: 9};

class Joseki extends React.Component {
  constructor(props) {
        super(props);
        this.state = {
          moves: "",
          board: new godash.Board(),
          nextColor: godash.BLACK,
          passNext: true,  // was 'tenuki' one of the next moves played here.
          other_highlights: defaultHighlights, // if so, what moves were considered then?
          highlights: defaultHighlights,
          search_enabled: false,
          chartData: null,
          tableData: null,
          count: null,
          run: null,
          tableHourSort: 'desc',
          tablePage: 1,
          doneMessage: false,
        };

        this.resetBoard = this.resetBoard.bind(this);
        this.coordinateClicked = this.coordinateClicked.bind(this);
        this.updateGraph = this.updateGraph.bind(this);
        this.findGames = this.findGames.bind(this);
        this.tenuki = this.tenuki.bind(this);
        this.handleRunChange = this.handleRunChange.bind(this);
        this.prevMove = this.prevMove.bind(this);
        this.handleKeyDown = this.handleKeyDown.bind(this);
        this.toggleHourSort = this.toggleHourSort.bind(this);
        this.handleNextButtonClick = this.handleNextButtonClick.bind(this);
        this.handlePrevButtonClick = this.handlePrevButtonClick.bind(this);

        this.updateHeatmap(); // update our heatmap with the empty board.
}

  handleKeyDown(key, e) {
    if (key === 'left') {
      this.prevMove(e);
    }
  }

  handleNextButtonClick() {
    this.setState(
        {tablePage: this.state.tablePage + 1},
        () => {this.findGames();}
    );
  }
  handlePrevButtonClick() {
    this.setState(
        {tablePage: this.state.tablePage - 1},
        () => {this.findGames();}
    );
  }

  resetBoard() {
    this.setState({
      board: new godash.Board(),
      nextColor: godash.BLACK,
      passNext: true,
      moves: "",
      highlights: null,
      other_highlights: null,
      chartData: null,
      tableData: null,
      tablePage: 1,
      search_enabled: false,
      run: null,
      doneMessage: false,
    }, () => {
      this.updateHeatmap();
    });
  }

  searchEnabled(moves) {
      return (moves.match(/;/g)||[]).length >= 3 ? true : false;
  }

  prevMove(event) {
      var num_moves = (this.state.moves.match(/;/g)||[]).length;
      if (num_moves <= 1){
        return;
      }
      var new_moves = this.state.moves.split(';')
      new_moves.pop()
      var m = new_moves.pop()
      new_moves = new_moves.join(';') + ';'
      this.setState({
        board: godash.removeStone(this.state.board, godash.sgfPointToCoordinate(m.slice(2,4))),
        nextColor: (this.state.nextColor === godash.BLACK ? godash.WHITE : godash.BLACK),
        moves: new_moves,
        search_enabled: this.searchEnabled(new_moves)
      }, () => { 
        this.updateHeatmap();
        if(num_moves >= 1) {
          this.updateGraph();
        }
      });

  }

  handleRunChange(event, newRun) {
    this.setState({
      run: newRun,
    }, () => {
      this.updateHeatmap();
      if (this.state.tableData) {
        this.findGames();
      }
    });
  }

  coordinateClicked(coordinate) {
      var m = this.state.nextColor === godash.BLACK ? "B[" : "W[";
      m = m + godash.coordinateToSgfPoint(coordinate) + "];";
      var num_moves = (this.state.moves.match(/;/g)||[]).length;

      this.setState({
          board: godash.addMove(this.state.board, coordinate, this.state.nextColor),
          nextColor: (this.state.nextColor === godash.BLACK ? godash.WHITE : godash.BLACK),
          moves: this.state.moves + m,
          highlights: null,
          search_enabled: this.searchEnabled(this.state.moves+m)
      }, () => {
        this.updateHeatmap();
        if(num_moves >= 1) {
          this.updateGraph();
        }
      });
  }

  updateHeatmap() {
      axios.post('/nexts', {
        params: {
          prefix: this.state.moves,
          run: this.state.run
        }
      }).then(response => {
        var next = this.state.nextColor === godash.BLACK ? 'W' : 'B';
        var highlights = new DefaultDict(Array);
        var other_highlights = new DefaultDict(Array);
        var passFound = false;
        for (const [coord,freq] of Object.entries(response.data.next_moves)) {
            if (coord[0] === next) {
              passFound = true;
              other_highlights[colors[Math.floor(freq * NUM_COLORS)]].push(godash.sgfPointToCoordinate(coord.slice(2,4)))
              continue;
            }
            highlights[colors[Math.floor(freq * NUM_COLORS)]].push(godash.sgfPointToCoordinate(coord.slice(2,4)))
        }
        var num_moves = (this.state.moves.match(/;/g)||[]).length;
        if (response.data.count === 0 && num_moves > 1) {
            this.setState({doneMessage: true});
            this.prevMove();
            this.findGames();
        } else {
          this.setState({
              highlights: highlights,
              other_highlights: other_highlights,
              passNext: passFound,
              count: response.data.count
          });
        }
      });
  }

  tenuki() {
      this.setState({
          nextColor: (this.state.nextColor === godash.BLACK ? godash.WHITE : godash.BLACK),
          other_highlights: this.state.highlights,
          highlights: this.state.other_highlights
      });
  }

  updateGraph() {
    axios.post('/search', {
      params: {
        sgf: this.state.moves,
      }
    }).then(response => {
      console.log(response.data);
      this.setState({
        chartData: [response.data.cols, ...response.data.rows]
      });

    });
  }

  toggleHourSort() {
    this.setState({
      tableHourSort: this.state.tableHourSort === 'desc' ? 'asc' : 'desc'
    }, () => {this.findGames();}
    );
  }

  findGames() {
    axios.post('/games', {
      params: {
        sgf: this.state.moves,
        run: this.state.run,
        sort: this.state.tableHourSort,
        page: this.state.tablePage,
      }
    }).then(response => {
      console.log(response.data);
      this.setState({
        tableData: response.data.rows
      });
    });
  }

  render() {
    var num_moves = (this.state.moves.match(/;/g)||[]).length;
    return (
  <ThemeProvider theme={theme}>
      <div className="App">
        <KeyboardEventHandler
            handleKeys={['left', 'right']}
                onKeyEvent={(key, e) => {
                  this.handleKeyDown(key, e);
                }} />

        <AppBar position="static">
          <Toolbar>
              <h2 align='left' className="{classes.title}">
                    {this.state.moves ? this.state.moves : "Joseki Explorer"}
              </h2>
          </Toolbar>
        </AppBar>
        <div style={{ marginTop:20 }}> </div>
        <Container>
              <Typography variant="h5" align='left' gutterBottom>
              <p>
              { this.state.moves ?
                <span> Seen: {this.state.count} times </span> : <span>&nbsp;</span>
              }
              { this.state.doneMessage === true ? <span> (no further data) </span> : <div> </div> }
              </p>
              </Typography>

          <Grid container justify="flex-start" spacing={3}>
            <Grid item md={5} xs={12}>
                <Goban
                    board={this.state.board}
                    onCoordinateClick={this.coordinateClicked}
                    highlights={this.state.highlights}
                    topLeft={topLeft}
                    bottomRight={bottomRight}
	    	    options= {{stonePadding: 2}}
                />
                <MyButton variant="contained" onClick={() => {
                  this.setState({ tablePage: 1 },
                      () => { this.findGames(); });
                }} color="primary" disabled={!this.state.search_enabled}>Search</MyButton>
                <MyButton variant="contained" onClick={this.tenuki}
                          color={this.state.passNext ? "secondary" : "default"}>Tenuki</MyButton>
                <MyButton variant="contained" onClick={this.resetBoard}>Clear</MyButton>
            {num_moves > 0 ? (
            <ToggleButtonGroup variant="contained" exclusive value={this.state.run} onChange={this.handleRunChange} size='small'>
                <ToggleButton variant="contained" value="v15">v15</ToggleButton>
                <ToggleButton variant="contained" value="v16">v16</ToggleButton>
                <ToggleButton variant="contained" value="v17">v17</ToggleButton>
            </ToggleButtonGroup>
            ) : <div></div>
            }
            </Grid>

            <Grid item md={7} xs={12} >
            {this.state.chartData === null ? (<div >
              <Typography variant="h5" align='left' gutterBottom>
              <p> Explore Minigo's most common opening moves during its training by clicking on the board to the left. </p>
              </Typography>
              <Typography variant="subtitle1" align='left' gutterBottom>
              <p> For sequences longer than two moves, a frequency graph will appear here showing the openings' popularity over time. </p>
              <p> Joseki's beginning with black or white are tabulated independently, as are transpositions.</p>
              <p> If tenuki was a frequently played option, 'tenuki' button will be enabled to toggle the next color to be played. </p>
              <p> "Search" will find hourly details and example games featuring the current pattern. </p>
              </Typography>
              </div>) :
                <Chart
                    loader={ <p> Chart loading... </p> }
                    chartType="ScatterChart"
                    data={this.state.chartData}
                    options = {{
                              title: `How frequently this sequence occurred, per hour, over training`,
                              hAxis: {title: '% of training',
                                      viewWindow: {min: 0, max: 100}},
                              vAxis: {title: 'Frequency', logScale: true},
                              chartArea: {'width': '80%', 'height': '80%'},
                              legend: { position : 'bottom'},
                              theme: 'material',
                              pointSize: 3,
                    }}
                    height={'600px'}
                />
            }
            </Grid>
            <Grid item xs={12}>
            {this.state.tableData === null ? <div> </div> :
              <Paper>
                <Typography id="tableTitle" variant="h5" component="div" align="left" style={{ padding:20 }} >
                  Details per hour and example games

                  <IconButton style={{marginLeft:30, marginRight:15}}
                              onClick={this.handlePrevButtonClick}
                              disabled={this.state.tablePage === 1}
                              aria-label="previous page">
                  <KeyboardArrowLeft/> </IconButton>
                  {this.state.tablePage}
                  <IconButton style={{marginRight:30, marginLeft: 15}}onClick={this.handleNextButtonClick} aria-label="next page">
                  <KeyboardArrowRight/> </IconButton>
                </Typography>

                <Table size="small" aria-labelledby="tableTitle">
                  <TableHead>
                    <TableRow>
                      <TableCell> Link </TableCell>
                      <TableCell>
                        <TableSortLabel
                            active={true}
                            direction={this.state.tableHourSort}
                            onClick={this.toggleHourSort} /> Hour </TableCell>
                      <TableCell> Run </TableCell>
                      <TableCell> Winrate </TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {this.state.tableData.map(row => (
                      <TableRow key={row.game}>
                        <TableCell scope="row">
                          <a href={"http://cloudygo.com/" + row.run + "-19x19/joseki/full/" + row.game}
                            target="_blank" rel="noopener noreferrer">{row.game}</a>
                        </TableCell>
                        <TableCell align="left"> {row.hour} </TableCell>
                        <TableCell align="left"> {row.run} </TableCell>
                        <TableCell align="left">
                        <div style={{backgroundColor: "#ccc", width:'100%', position:'relative'}}>
                          <div style={{borderRight: '2px dashed #eee',
                                       width:'50%',
                                       position: 'absolute'}}> &nbsp; </div>
                          <div style={{backgroundColor: "#111", width:(row.winrate*100) + "%"}}> &nbsp; </div>
                        </div> </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </Paper>
            }
            </Grid>
          </Grid>
        </Container>
      </div>
  </ThemeProvider>
    );
  }
}

function App() {
  return ( <Joseki /> );
}

export default App;
