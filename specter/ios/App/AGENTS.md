# ios/App/AGENTS.md

- Function summarizes actions, not how things are done.
Bad:
```
func makeEnrichedParams(input: Input) -> Result<Params, Error> {
    let params = Params(
        name: input.name,
        surname: input.surname
    )
    if input.isFemale {
        params.age = .unavailable
    }

    if params.shouldBeEnriched {
        return .success(enrich(params))
    }

    return .failure(NSError(""))
}
```

Good:
```
extension Input {
    fileprivate var params: Params {
        modify(Params(
            name: input.name,
            surname: input.surname
        )) {
            $0.age = $0.isFemale ? .unvailable : .available(age)
        }
    }
}
func makeEnrichedParams(input: Input) -> Params {
    let params = input.params
    if params.shouldBeEnriched {
        return .success(enrich(params))
    }
    return .failure(NSError(""))
}
```

2. Use computed properties instead of stored when possible
3. Use struct over protocols:
```
protocol A {
    func doSome() {}
}

struct A {
    var doSome: () -> Void
}
```

