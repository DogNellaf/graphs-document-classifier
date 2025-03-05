graph [
  directed 1
  multigraph 1
  node [
    id 0
    label "say-01"
  ]
  node [
    id 1
    label "person"
  ]
  node [
    id 2
    label "recommend-01"
  ]
  node [
    id 3
    label "post-01"
  ]
  node [
    id 4
    label "previous"
  ]
  node [
    id 5
    label "seek-01"
  ]
  node [
    id 6
    label "one"
  ]
  node [
    id 7
    label "assist-01"
  ]
  node [
    id 8
    label "inject-01"
  ]
  node [
    id 9
    label "have-rel-role-91"
  ]
  node [
    id 10
    label "doctor"
  ]
  node [
    id 11
    label "contrast-01"
  ]
  node [
    id 12
    label "amr-unknown"
  ]
  node [
    id 13
    label "product"
  ]
  node [
    id 14
    label "name"
  ]
  node [
    id 15
    label "sumatriptin"
  ]
  node [
    id 16
    label "obligate-01"
  ]
  node [
    id 17
    label "-"
  ]
  node [
    id 18
    label "immediate"
  ]
  node [
    id 19
    label "onset"
  ]
  node [
    id 20
    label "migraine"
  ]
  edge [
    source 0
    target 1
    key 0
    label ":ARG0"
  ]
  edge [
    source 0
    target 2
    key 0
    label ":ARG1"
  ]
  edge [
    source 2
    target 5
    key 0
    label ":ARG1"
  ]
  edge [
    source 2
    target 6
    key 0
    label ":ARG2"
  ]
  edge [
    source 3
    target 1
    key 0
    label ":ARG0"
  ]
  edge [
    source 3
    target 4
    key 0
    label ":time"
  ]
  edge [
    source 5
    target 6
    key 0
    label ":ARG0"
  ]
  edge [
    source 5
    target 7
    key 0
    label ":ARG1"
  ]
  edge [
    source 7
    target 1
    key 0
    label ":ARG0"
  ]
  edge [
    source 7
    target 6
    key 0
    label ":ARG1"
  ]
  edge [
    source 7
    target 8
    key 0
    label ":ARG2"
  ]
  edge [
    source 8
    target 6
    key 0
    label ":ARG0"
  ]
  edge [
    source 8
    target 6
    key 1
    label ":ARG2"
  ]
  edge [
    source 8
    target 18
    key 0
    label ":time"
  ]
  edge [
    source 9
    target 1
    key 0
    label ":ARG0"
  ]
  edge [
    source 9
    target 6
    key 0
    label ":ARG1"
  ]
  edge [
    source 9
    target 10
    key 0
    label ":ARG2"
  ]
  edge [
    source 11
    target 12
    key 0
    label ":ARG2"
  ]
  edge [
    source 12
    target 13
    key 0
    label ":topic"
  ]
  edge [
    source 13
    target 14
    key 0
    label ":name"
  ]
  edge [
    source 14
    target 15
    key 0
    label ":op1"
  ]
  edge [
    source 16
    target 17
    key 0
    label ":polarity"
  ]
  edge [
    source 16
    target 6
    key 0
    label ":ARG1"
  ]
  edge [
    source 16
    target 8
    key 0
    label ":ARG2"
  ]
  edge [
    source 16
    target 12
    key 0
    label ":polarity"
  ]
  edge [
    source 18
    target 19
    key 0
    label ":op1"
  ]
  edge [
    source 19
    target 20
    key 0
    label ":op1"
  ]
]
