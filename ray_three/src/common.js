
function modify(object, callback) {
    let obj = object;
    callback(obj);
    return obj;
}

export { modify };